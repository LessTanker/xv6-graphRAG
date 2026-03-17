import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
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
    from backend.core.CommunityManager import CommunityManager
except ImportError:
    import config  # type: ignore
    import utils  # type: ignore
    from core.CommunityManager import CommunityManager  # type: ignore


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

    def ensure_ready(self, force_rebuild: bool = False) -> None:
        required_paths = [
            config.CHUNKS_METADATA_PATH,
            config.GRAPH_EDGES_PATH,
            config.FAISS_INDEX_PATH,
        ]
        if force_rebuild or any(not p.exists() for p in required_paths):
            self.build_all()

        if force_rebuild or not config.COMMUNITIES_PATH.exists():
            manager = CommunityManager(
                edges_path=config.GRAPH_EDGES_PATH,
                chunks_path=config.CHUNKS_METADATA_PATH,
                output_path=config.COMMUNITIES_PATH,
                prefer_static_partition=config.COMMUNITY_USE_STATIC_PATH_PARTITION,
            )
            manager.run_leiden_algorithm()
            manager.summarize_communities()
            manager.save_communities()

        if force_rebuild or not config.COMMUNITY_FAISS_INDEX_PATH.exists():
            self._build_and_save_community_embeddings()

        if force_rebuild or not config.EXPERT_PATHS_PATH.exists():
            self._build_and_save_expert_paths()

    def build_all(self) -> None:
        compile_db = utils.load_json_array(config.COMPILE_COMMANDS_PATH, "compile_commands.json")
        compile_db = self._ensure_user_compile_entries(compile_db)
        chunks = self._extract_chunks(compile_db)
        edges = self._build_edges(chunks, compile_db)
        self._build_and_save_embeddings(chunks)
        utils.save_json(config.CHUNKS_METADATA_PATH, chunks)
        utils.save_json(config.GRAPH_EDGES_PATH, {"edges": edges})

    def _ensure_user_compile_entries(self, compile_db: List[Dict]) -> List[Dict]:
        """Ensure user/*.c translation units exist so syscall transition edges can be extracted."""
        user_dir = self.source_root / "xv6-riscv" / "user"
        if not user_dir.exists():
            return compile_db

        augmented = list(compile_db)
        existing_user_files: Set[str] = set()

        for entry in compile_db:
            src_file = str(entry.get("file", "")).strip()
            if not src_file.endswith(".c"):
                continue
            file_abs, directory_abs = self._resolve_entry_file(entry)
            if file_abs is None or directory_abs is None:
                continue
            rel_file = utils.to_rel_xv6(str(file_abs))
            if rel_file and rel_file.startswith("user/"):
                existing_user_files.add(rel_file)

        base_directory = self.source_root / "xv6-riscv"
        for c_file in sorted(user_dir.glob("*.c")):
            rel_file = f"user/{c_file.name}"
            if rel_file in existing_user_files:
                continue

            augmented.append(
                {
                    "directory": str(base_directory),
                    "file": rel_file,
                    "arguments": [
                        "clang",
                        "-I.",
                        "-I./kernel",
                        "-I./user",
                        "-x",
                        "c",
                        "-std=c11",
                        "-ffreestanding",
                        "-fno-builtin",
                        "-c",
                        rel_file,
                    ],
                }
            )

        return augmented

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
                code_compact = self._compact(raw_code, elem_type)
                dedup_key = (elem_type, name, rel_file, code_compact)
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

    def _build_and_save_embeddings(self, chunks: List[Dict]) -> None:
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

    def _build_and_save_expert_paths(self) -> None:
        chunks = utils.load_metadata(config.CHUNKS_METADATA_PATH)
        edges = utils.load_edges(config.GRAPH_EDGES_PATH)
        selected = self._extract_expert_paths_with_llm(chunks, edges)

        payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "path_count": len(selected),
            "expert_paths": selected,
        }
        utils.save_json(config.EXPERT_PATHS_PATH, payload)

    def _extract_expert_paths_with_llm(self, chunks: List[Dict], edges: List[Dict]) -> List[Dict[str, object]]:
        min_chain_len = 3
        function_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("type") == "function" and isinstance(chunk.get("id"), int)
        ]
        if not function_chunks:
            return []

        call_edges: Set[Tuple[int, int]] = set()
        out_degree = Counter()
        in_degree = Counter()
        call_out: DefaultDict[int, List[int]] = defaultdict(list)
        call_in: DefaultDict[int, List[int]] = defaultdict(list)
        for edge in edges:
            if str(edge.get("type", "")).strip() != REL_CALLS:
                continue
            src = edge.get("from")
            dst = edge.get("to")
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            call_edges.add((src, dst))
            out_degree[src] += 1
            in_degree[dst] += 1
            call_out[src].append(dst)
            call_in[dst].append(src)

        name_to_nodes: DefaultDict[str, List[Dict]] = defaultdict(list)
        by_id: Dict[int, Dict] = {}
        for chunk in function_chunks:
            cid = int(chunk["id"])
            by_id[cid] = chunk
            name_to_nodes[str(chunk.get("name", "")).strip().lower()].append(chunk)

        catalog_items: List[Dict[str, str]] = []
        for chunk in function_chunks:
            cid = int(chunk["id"])
            name = str(chunk.get("name", "")).strip()
            file_name = str(chunk.get("file", "")).strip()
            summary = str(chunk.get("summary", "")).strip()
            score = out_degree.get(cid, 0) + in_degree.get(cid, 0)
            top_callees = [str(by_id[nid].get("name", "")) for nid in call_out.get(cid, [])[:3] if nid in by_id]
            top_callers = [str(by_id[nid].get("name", "")) for nid in call_in.get(cid, [])[:3] if nid in by_id]
            catalog_items.append(
                {
                    "id": str(cid),
                    "name": name,
                    "file": file_name,
                    "summary": summary[:160],
                    "callees": ",".join(top_callees),
                    "callers": ",".join(top_callers),
                    "out": str(out_degree.get(cid, 0)),
                    "in": str(in_degree.get(cid, 0)),
                    "score": str(score),
                }
            )

        # Derive from the full graph/chunks set, then send a compact but broad catalog to LLM.
        catalog_items.sort(
            key=lambda item: (
                int(item["score"]),
                1 if str(item["file"]).startswith("kernel/") else 0,
                item["name"],
            ),
            reverse=True,
        )
        catalog_items = catalog_items[:420]

        max_count = max(1, int(config.EXPERT_PATH_MAX_COUNT))
        prompt_lines = [
            "You are an xv6 kernel expert. The following catalog is derived from chunks_metadata.json and graph_edges.json.",
            f"Output between 12 and {max_count} chains. Chain length can vary (prefer 3-8 functions).",
            "Important: do NOT fill with boring utility tails (panic/printf/release/push_off/pop_off unless absolutely central).",
            "Coverage must include: scheduler/context switch, process lifecycle (fork/exec/wait/exit), trap/syscall path, memory management, and filesystem I/O.",
            "Each chain must be coherent and focused on a kernel behavior.",
            "Return JSON only: {\"paths\":[{\"title\":\"...\",\"description\":\"...\",\"node_ids\":[1,2,3],\"functions\":[\"f1\",\"f2\",...]}]}",
            "Prefer node_ids because names may be ambiguous.",
            "Function catalog:",
        ]
        for idx, item in enumerate(catalog_items, 1):
            prompt_lines.append(
                f"{idx}. id={item['id']} {item['name']} ({item['file']}) score={item['score']} in={item['in']} out={item['out']} "
                f"callers=[{item['callers']}] callees=[{item['callees']}] summary={item['summary']}"
            )

        raw = utils.call_llm_api(
            query="From the catalog, identify core xv6 call chains and function order.",
            context_markdown="\n".join(prompt_lines),
            response_language="English",
        )

        parsed = self._parse_json_object(raw)
        llm_paths = parsed.get("paths", []) if isinstance(parsed, dict) else []
        if not isinstance(llm_paths, list):
            llm_paths = []

        utility_names = {
            "panic",
            "printf",
            "printint",
            "release",
            "acquire",
            "push_off",
            "pop_off",
            "intr_on",
            "intr_off",
            "intr_get",
            "mycpu",
            "cpuid",
            "holding",
            "holdingsleep",
        }

        def pick_node_by_name(name: str) -> Optional[Dict]:
            candidates = name_to_nodes.get(name.strip().lower(), [])
            if not candidates:
                return None
            candidates = sorted(
                candidates,
                key=lambda c: (
                    1 if str(c.get("file", "")).startswith("kernel/") else 0,
                    out_degree.get(int(c.get("id", -1)), 0),
                ),
                reverse=True,
            )
            return candidates[0]

        def trim_utility_tail(ids: List[int]) -> List[int]:
            trimmed = list(ids)
            while len(trimmed) > min_chain_len:
                tail_name = str(by_id.get(trimmed[-1], {}).get("name", "")).strip().lower()
                if tail_name in utility_names:
                    trimmed.pop()
                    continue
                break
            return trimmed

        def bridge_one_hop(a: int, b: int) -> Optional[int]:
            # Try repairing one missing edge with a meaningful intermediate node.
            for mid in call_out.get(a, []):
                if (mid, b) not in call_edges:
                    continue
                mid_name = str(by_id.get(mid, {}).get("name", "")).strip().lower()
                if mid_name in utility_names:
                    continue
                return mid
            return None

        def ensure_connected(ids: List[int]) -> List[int]:
            if len(ids) < 2:
                return ids
            repaired = [ids[0]]
            for nxt in ids[1:]:
                prev = repaired[-1]
                if (prev, nxt) in call_edges:
                    repaired.append(nxt)
                    continue
                bridge = bridge_one_hop(prev, nxt)
                if bridge is not None and bridge not in repaired:
                    repaired.append(bridge)
                    repaired.append(nxt)
                    continue
                # keep chain coherent: stop when jump cannot be repaired
                break
            return repaired

        def longest_connected_segment(ids: List[int]) -> List[int]:
            if len(ids) < 2:
                return ids
            best: List[int] = []
            cur: List[int] = [ids[0]]
            for nxt in ids[1:]:
                if (cur[-1], nxt) in call_edges:
                    cur.append(nxt)
                else:
                    if len(cur) > len(best):
                        best = cur[:]
                    cur = [nxt]
            if len(cur) > len(best):
                best = cur[:]
            return best if best else ids

        def detect_themes(names: List[str]) -> List[str]:
            lowered = [n.lower() for n in names]
            themes = []
            if any(k in " ".join(lowered) for k in ["scheduler", "sched", "swtch"]):
                themes.append("scheduler")
            if any(k in " ".join(lowered) for k in ["fork", "exec", "wait", "exit", "proc"]):
                themes.append("process")
            if any(k in " ".join(lowered) for k in ["sys_", "syscall", "trap", "intr"]):
                themes.append("syscall_trap")
            if any(k in " ".join(lowered) for k in ["kalloc", "kfree", "uvm", "walk", "copyin", "copyout", "vm"]):
                themes.append("memory")
            if any(k in " ".join(lowered) for k in ["namex", "inode", "ilock", "bread", "log", "file", "pipe", "sys_open", "sys_unlink", "sys_link"]):
                themes.append("filesystem")
            return sorted(set(themes))

        output: List[Dict[str, object]] = []
        seen: Set[Tuple[int, ...]] = set()

        def ingest_paths(paths: List[Dict[str, object]]) -> None:
            for item in paths:
                if len(output) >= max_count:
                    break
                if not isinstance(item, dict):
                    continue
                ids: List[int] = []
                raw_ids = item.get("node_ids", [])
                if isinstance(raw_ids, list):
                    for nid in raw_ids:
                        if not isinstance(nid, int):
                            continue
                        if nid not in by_id:
                            continue
                        if ids and ids[-1] == nid:
                            continue
                        ids.append(nid)

                if len(ids) < 4:
                    funcs = item.get("functions", [])
                    if isinstance(funcs, list):
                        for name in funcs:
                            node = pick_node_by_name(str(name))
                            if node is None:
                                continue
                            node_id = int(node["id"])
                            if ids and ids[-1] == node_id:
                                continue
                            ids.append(node_id)

                if len(ids) < min_chain_len:
                    continue
                ids = ensure_connected(ids)
                ids = longest_connected_segment(ids)
                ids = trim_utility_tail(ids)
                if len(ids) < min_chain_len:
                    continue

                key = tuple(ids)
                if key in seen:
                    continue
                seen.add(key)

                names = [str(by_id[nid].get("name", "")) for nid in ids if nid in by_id]
                themes = detect_themes(names)
                if not themes:
                    continue

                output.append(
                    {
                        "id": f"expert_{len(output) + 1}",
                        "title": str(item.get("title", "")).strip() or " -> ".join(names),
                        "description": str(item.get("description", "")).strip(),
                        "node_ids": ids,
                        "node_names": names,
                        "edge_types": [REL_CALLS] * (len(ids) - 1),
                        "source": "llm_inferred",
                        "themes": themes,
                    }
                )

        ingest_paths(llm_paths)

        # LLM-only second pass for additional unique chains (no deterministic fallback).
        if len(output) < max_count:
            existing = [" -> ".join(item.get("node_names", [])) for item in output]
            extra_prompt = [
                "Generate additional unique xv6 core chains not overlapping with existing ones.",
                f"Need up to {max_count - len(output)} more chains.",
                "Return JSON only with key paths, same schema as before.",
                f"Existing chains to avoid: {existing}",
                "Function catalog:",
            ]
            for idx, item in enumerate(catalog_items, 1):
                extra_prompt.append(
                    f"{idx}. id={item['id']} {item['name']} ({item['file']}) score={item['score']} in={item['in']} out={item['out']} "
                    f"callers=[{item['callers']}] callees=[{item['callees']}] summary={item['summary']}"
                )
            extra_raw = utils.call_llm_api(
                query="Provide additional unique core xv6 chains.",
                context_markdown="\n".join(extra_prompt),
                response_language="English",
            )
            extra_parsed = self._parse_json_object(extra_raw)
            extra_paths = extra_parsed.get("paths", []) if isinstance(extra_parsed, dict) else []
            if isinstance(extra_paths, list):
                ingest_paths(extra_paths)

        return output[:max_count]

    def _parse_json_object(self, text: str) -> Optional[Dict[str, object]]:
        candidate = str(text or "").strip()
        if not candidate:
            return None
        if "```json" in candidate:
            candidate = candidate.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in candidate:
            candidate = candidate.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(candidate[start : end + 1])
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
        return None

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

    def _iter_descendants(self, node) -> Iterable:
        for child in node.get_children():
            yield child
            yield from self._iter_descendants(child)

    def _read_lines(self, file_path: Path) -> Optional[List[str]]:
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
                return fh.read().splitlines()
        except OSError:
            return None

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

    def _compact(self, code: str, elem_type: str) -> str:
        code = code.strip()
        if elem_type in {"function", "struct"}:
            brace_pos = code.find("{")
            if brace_pos != -1:
                return code[:brace_pos].rstrip() + " { ... }"
        return code

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
