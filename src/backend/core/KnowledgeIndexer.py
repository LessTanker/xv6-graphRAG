import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import clang.cindex as cindex
except ImportError as exc:
    raise SystemExit("libclang bindings are missing. Install with: pip install libclang") from exc

try:
    from backend import config, utils
    from backend.core import CommunityManager, ExpertPathManager
except ImportError:
    import config  # type: ignore
    import utils  # type: ignore
    from core import CommunityManager, ExpertPathManager  # type: ignore


TARGET_TYPES = {"function", "struct", "global_var", "macro"}
REL_CALLS = "CALLS"
REL_USES_STRUCT = "USES_STRUCT"
REL_USES_GLOBAL = "USES_GLOBAL"
REL_MACRO_USE = "MACRO_USE"
REL_USER_TO_KERNEL = "USER_TO_KERNEL"
REL_KERNEL_TO_USER = "KERNEL_TO_USER"


class KnowledgeIndexer:
    """Offline pipeline: parse code, build graph, and create embeddings/index."""

    def __init__(self, source_root: Path, model: Optional[SentenceTransformer] = None):
        self.source_root = Path(source_root)
        self.model = model or SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self._clang_index = cindex.Index.create()

    # Check if all required indexing and analysis artifacts exist on disk.
    # If any are missing or force_rebuild is True, trigger the necessary 
    # build or analysis pipelines.
    def ensure_ready(self, force_rebuild: bool = False) -> None:
        required_paths = [
            config.CHUNKS_METADATA_PATH,
            config.GRAPH_EDGES_PATH,
            config.FAISS_INDEX_PATH,
            config.COMMUNITIES_PATH,
            config.COMMUNITY_FAISS_INDEX_PATH,
            config.EXPERT_PATHS_PATH,
        ]
        if force_rebuild or any(not p.exists() for p in required_paths):
            self.build_all()

    # Orchestrate the complete offline indexing and analysis pipeline.
    def build_all(self) -> None:
        compile_db = utils.load_compile_db(config.COMPILE_COMMANDS_PATH, self.source_root)
        chunks = self._extract_chunks(compile_db)
        edges = self._build_edges(chunks, compile_db)
        
        utils.save_json(config.CHUNKS_METADATA_PATH, chunks)
        utils.save_json(config.GRAPH_EDGES_PATH, {"edges": edges})
        
        self._build_communities()
        self._build_expert_paths()
        
        self._build_and_save_chunks_embeddings()
        self._build_and_save_community_embeddings()

    # Run community detection and generate summaries using LLM.
    def _build_communities(self) -> None:
        manager = CommunityManager(
            edges_path=config.GRAPH_EDGES_PATH,
            chunks_path=config.CHUNKS_METADATA_PATH,
            output_path=config.COMMUNITIES_PATH,
            prefer_static_partition=config.COMMUNITY_USE_STATIC_PATH_PARTITION,
        )
        manager.run_leiden_algorithm()
        manager.summarize_communities()
        manager.save_communities()

    # Extract and analyze core kernel call chains using LLM.
    def _build_expert_paths(self) -> None:
        manager = ExpertPathManager(
            chunks_path=config.CHUNKS_METADATA_PATH,
            edges_path=config.GRAPH_EDGES_PATH,
            output_path=config.EXPERT_PATHS_PATH,
        )
        manager.build_and_save()

    # Vectorize code chunks and build FAISS index for semantic retrieval.
    def _build_and_save_chunks_embeddings(self) -> None:
        chunks = utils.load_metadata(config.CHUNKS_METADATA_PATH)
        if not chunks:
            raise SystemExit("No chunks extracted; cannot build FAISS index.")

        texts = [utils.chunk_to_text(chunk) for chunk in chunks]
        embeddings = self.model.encode(texts, batch_size=max(1, config.EMBEDDING_BATCH_SIZE), convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))

        chunk_hashes = {
            str(chunk["id"]): hashlib.sha256(text.encode("utf-8")).hexdigest()
            for chunk, text in zip(chunks, texts)
            if chunk.get("id") is not None
        }
        utils.save_json(
            config.EMBEDDING_STATE_PATH,
            {
                "model_name": config.EMBEDDING_MODEL_NAME,
                "chunk_count": len(chunks),
                "chunk_hashes": chunk_hashes,
            },
        )
        np.savez_compressed(
            str(config.EMBEDDINGS_CACHE_PATH),
            ids=np.array([int(chunk["id"]) for chunk in chunks], dtype=np.int64),
            embeddings=embeddings,
        )

    # Vectorize community summaries and build FAISS index for high-level conceptual retrieval.
    def _build_and_save_community_embeddings(self) -> None:
        data = utils.load_json_object(config.COMMUNITIES_PATH, "communities.json")
        community_nodes = data.get("Community_Nodes", [])
        if not isinstance(community_nodes, list) or not community_nodes:
            raise SystemExit("No Community_Nodes found; cannot build community FAISS index.")

        community_ids: List[str] = []
        texts: List[str] = []
        for item in community_nodes:
            if not isinstance(item, dict):
                continue
            cid = item.get("community_id")
            summary = str(item.get("summary", "")).strip()
            if cid is None or not summary:
                continue
            community_ids.append(str(cid))
            texts.append(summary)

        if not texts:
            raise SystemExit("Community summaries are empty; cannot build community FAISS index.")

        embeddings = self.model.encode(texts, batch_size=max(1, config.EMBEDDING_BATCH_SIZE), convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, str(config.COMMUNITY_FAISS_INDEX_PATH))

        np.savez_compressed(
            str(config.COMMUNITY_EMBEDDINGS_CACHE_PATH),
            community_ids=np.array([int(cid) if str(cid).isdigit() else -1 for cid in community_ids], dtype=np.int64),
            embeddings=embeddings,
        )

    # Extract code elements (functions, structs, globals, macros) from the source files listed in the compilation database.
    # For each element, deduplicate using a key of (type, name, file, code hash) to avoid collecting identical code blocks multiple times.
    # Returns a list of unique code element dictionaries with metadata and source code.
    def _extract_chunks(self, compile_db: List[Dict]) -> List[Dict]:
        elements: List[Dict] = []
        seen: Set[Tuple[str, str, str, str]] = set()
        next_id = 1

        for entry in compile_db:
            src_file = entry.get("file")
            if not src_file or not str(src_file).endswith((".c", ".h")):
                continue

            file_abs, directory_abs = self._resolve_entry_file(entry)
            if file_abs is None:
                continue

            args = utils.sanitize_compile_args(entry.get("arguments", []), str(src_file), str(directory_abs))
            if not args:
                continue

            tu = self._parse_tu(str(file_abs), args)
            if tu is None:
                continue

            for node in self._iter_descendants(tu.cursor):
                elem_type = self._cursor_to_element_type(node)
                if elem_type is None:
                    continue
                if not node.location.file:
                    continue

                node_file = Path(os.path.normpath(node.location.file.name))
                rel_file = utils.to_rel_xv6(str(node_file))
                if rel_file is None:
                    continue

                source_lines = self._read_lines(node_file)
                if source_lines is None:
                    continue

                name = (node.spelling or "").strip()
                if not name:
                    continue

                raw_code = self._source_range(node, source_lines)
                if not raw_code.strip():
                    continue

                raw_code = raw_code.strip()
                code_hash = hashlib.md5(raw_code.encode()).hexdigest()
                dedup_key = (elem_type, name, rel_file, code_hash)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                elements.append(
                    {
                        "id": next_id,
                        "type": elem_type,
                        "name": name,
                        "file": rel_file,
                        "summary": "",
                        "keywords": [],
                        "code": raw_code,
                    }
                )
                next_id += 1

        return elements

    def _build_edges(self, chunks: List[Dict], compile_db: List[Dict]) -> List[Dict]:
        edges: Set[Tuple[int, int, str]] = set()
        (
            funcs_by_name,
            funcs_by_file_name,
            structs_by_name,
            globals_by_name,
            macros_by_name,
        ) = self._build_name_maps(chunks)

        for entry in compile_db:
            src_file = entry.get("file")
            if not src_file or not str(src_file).endswith(".c"):
                continue

            file_abs, directory_abs = self._resolve_entry_file(entry)
            if file_abs is None:
                continue

            rel_file = utils.to_rel_xv6(str(file_abs))
            if rel_file is None:
                continue

            args = utils.sanitize_compile_args(entry.get("arguments", []), str(src_file), str(directory_abs))
            if not args:
                continue

            tu = self._parse_tu(str(file_abs), args)
            if tu is None:
                continue

            macro_lines = self._extract_macro_lines(tu, str(file_abs))

            for cur in tu.cursor.get_children():
                if cur.kind != cindex.CursorKind.FUNCTION_DECL or not cur.is_definition():
                    continue
                if not cur.location.file:
                    continue
                if os.path.normpath(cur.location.file.name) != os.path.normpath(str(file_abs)):
                    continue

                src_name = cur.spelling
                src_id = funcs_by_file_name.get((rel_file, src_name))
                if src_id is None:
                    continue

                start_line = cur.extent.start.line
                end_line = cur.extent.end.line
                for line_no, macro_name in macro_lines:
                    if start_line <= line_no <= end_line:
                        for dst_id in macros_by_name.get(macro_name, set()):
                            self._add_edge(edges, src_id, dst_id, REL_MACRO_USE)

                for node in self._iter_descendants(cur):
                    if node.kind == cindex.CursorKind.CALL_EXPR:
                        callee = node.referenced.spelling if node.referenced else node.spelling
                        if callee:
                            for dst_id in funcs_by_name.get(callee, set()):
                                self._add_edge(edges, src_id, dst_id, REL_CALLS)

                            # User-mode call sites that target syscall wrappers are linked
                            # directly to kernel sys_* handlers as transition edges.
                            if rel_file.startswith("user/"):
                                for dst_id in funcs_by_name.get(f"sys_{callee}", set()):
                                    self._add_edge(edges, src_id, dst_id, REL_USER_TO_KERNEL)

                                # Also capture the trap entry transition to syscall dispatcher.
                                for dst_id in funcs_by_name.get("syscall", set()):
                                    self._add_edge(edges, src_id, dst_id, REL_USER_TO_KERNEL)

                    elif node.kind == cindex.CursorKind.TYPE_REF:
                        ref = node.referenced
                        if ref and ref.kind == cindex.CursorKind.STRUCT_DECL and ref.spelling:
                            for dst_id in structs_by_name.get(ref.spelling, set()):
                                self._add_edge(edges, src_id, dst_id, REL_USES_STRUCT)

                    elif node.kind == cindex.CursorKind.DECL_REF_EXPR:
                        ref = node.referenced
                        if ref and ref.kind == cindex.CursorKind.VAR_DECL:
                            parent = ref.semantic_parent
                            if parent and parent.kind == cindex.CursorKind.TRANSLATION_UNIT:
                                for dst_id in globals_by_name.get(ref.spelling, set()):
                                    self._add_edge(edges, src_id, dst_id, REL_USES_GLOBAL)

        # Kernel-mode return path back to user mode is modeled explicitly.
        return_targets = set(funcs_by_name.get("usertrapret", set()))
        if not return_targets:
            return_targets = set(funcs_by_name.get("userret", set()))

        if return_targets:
            for func_name, src_ids in funcs_by_name.items():
                if not func_name.startswith("sys_") and func_name != "syscall":
                    continue
                for src_id in src_ids:
                    for dst_id in return_targets:
                        self._add_edge(edges, src_id, dst_id, REL_KERNEL_TO_USER)

        return [
            {"from": src, "to": dst, "type": rel}
            for src, dst, rel in sorted(edges, key=lambda x: (x[2], x[0], x[1]))
        ]

    # Resolve relative paths from a compilation database entry into absolute system paths.
    # Returns a tuple of (absolute_file_path, absolute_working_directory).
    def _resolve_entry_file(self, entry: Dict) -> Tuple[Optional[Path], Optional[Path]]:
        src_file = entry.get("file")
        directory = entry.get("directory", ".")

        directory_abs = Path(directory)
        if not directory_abs.is_absolute():
            directory_abs = (self.source_root / directory_abs).resolve()

        file_abs = Path(src_file)
        if not file_abs.is_absolute():
            file_abs = (self.source_root / file_abs).resolve()
        if not file_abs.exists():
            file_abs = (directory_abs / str(src_file)).resolve()
        if not file_abs.exists():
            return None, None
        return file_abs, directory_abs

    # Parse a C source file into a Clang TranslationUnit (AST root).
    # Tries precise parsing with compile_commands.json arguments; if that fails, falls back to generic kernel/user include paths.
    # Returns a TranslationUnit object on success, or None if parsing fails.
    def _parse_tu(self, file_abs: str, args: List[str]):
        options = cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        try:
            return self._clang_index.parse(path=file_abs, args=args, options=options)
        except Exception:
            fallback_args = [
                "-x",
                "c",
                "-std=c11",
                "-ffreestanding",
                "-fno-builtin",
                "-I",
                str(self.source_root / "xv6-riscv"),
                "-I",
                str(self.source_root / "xv6-riscv" / "kernel"),
                "-I",
                str(self.source_root / "xv6-riscv" / "user"),
            ]
            try:
                return self._clang_index.parse(path=file_abs, args=fallback_args, options=options)
            except Exception:
                return None
            
    # Recursively yield all descendant nodes of an AST node (depth-first traversal).
    def _iter_descendants(self, node) -> Iterable:
        for child in node.get_children():
            yield child
            yield from self._iter_descendants(child)

    # Determine the type of a given AST node.
    # Returns "function", "struct", "global_var", or "macro" for recognized code elements,
    # or None if the node is not a top-level chunk of interest.
    def _cursor_to_element_type(self, node) -> Optional[str]:
        if node.kind == cindex.CursorKind.FUNCTION_DECL and node.is_definition():
            return "function"
        if node.kind == cindex.CursorKind.STRUCT_DECL and node.is_definition() and node.spelling:
            return "struct"
        if node.kind == cindex.CursorKind.VAR_DECL:
            parent = node.semantic_parent
            if parent and parent.kind == cindex.CursorKind.TRANSLATION_UNIT:
                return "global_var"
        if node.kind == cindex.CursorKind.MACRO_DEFINITION:
            return "macro"
        return None

    # Read the entire content of a file and return it as a list of lines (strings).
    # Returns None if the file cannot be opened or read (e.g., file not found or permission error).
    def _read_lines(self, file_path: Path) -> Optional[List[str]]:
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
                return fh.read().splitlines()
        except OSError:
            return None

    # Extract the exact source code text for an AST node from the file's lines, using the node's start and end positions.
    # Handles both single-line and multi-line nodes, returning the code as a string. Returns an empty string if the range is invalid.
    def _source_range(self, node, source_lines: List[str]) -> str:
        start = node.extent.start
        end = node.extent.end
        if start.line <= 0 or end.line <= 0:
            return ""

        sl, sc = start.line - 1, max(0, start.column - 1)
        el, ec = end.line - 1, max(0, end.column - 1)
        lines = source_lines[sl : el + 1]
        if not lines:
            return ""
        if len(lines) == 1:
            return lines[0][sc:ec]
        return "\n".join([lines[0][sc:]] + lines[1:-1] + [lines[-1][:ec]])

    # Build lookup maps for code elements by name and file.
    # Groups element ids by type (function, struct, global_var, macro),
    # and provides fast access to ids via name or (file, name) tuple.
    # Used for relationship analysis and quick element retrieval.
    def _build_name_maps(self, nodes: List[Dict]):
        funcs_by_name: DefaultDict[str, Set[int]] = defaultdict(set)
        funcs_by_file_name: Dict[Tuple[str, str], int] = {}
        structs_by_name: DefaultDict[str, Set[int]] = defaultdict(set)
        globals_by_name: DefaultDict[str, Set[int]] = defaultdict(set)
        macros_by_name: DefaultDict[str, Set[int]] = defaultdict(set)

        for node in nodes:
            node_id = node.get("id")
            name = node.get("name", "")
            rel_file = node.get("file", "")
            typ = node.get("type")
            if node_id is None or not name:
                continue

            if typ == "function":
                funcs_by_name[name].add(node_id)
                if rel_file:
                    funcs_by_file_name[(rel_file, name)] = node_id
            elif typ == "struct":
                structs_by_name[name].add(node_id)
            elif typ == "global_var":
                globals_by_name[name].add(node_id)
            elif typ == "macro":
                macros_by_name[name].add(node_id)

        return funcs_by_name, funcs_by_file_name, structs_by_name, globals_by_name, macros_by_name

    # Extract macro instantiation locations and names from a translation unit.
    # Returns a list of (line number, macro name) tuples for macros instantiated in the given file.
    def _extract_macro_lines(self, tu, file_abs: str) -> List[Tuple[int, str]]:
        items: List[Tuple[int, str]] = []
        for cur in tu.cursor.get_children():
            if cur.kind != cindex.CursorKind.MACRO_INSTANTIATION:
                continue
            if not cur.location.file:
                continue
            if os.path.normpath(cur.location.file.name) != os.path.normpath(file_abs):
                continue
            if cur.spelling:
                items.append((cur.location.line, cur.spelling))
        return items

    def _add_edge(self, edge_set: Set[Tuple[int, int, str]], src: int, dst: int, rel: str) -> None:
        if src == dst:
            return
        edge_set.add((src, dst, rel))
