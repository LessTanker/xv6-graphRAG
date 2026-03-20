# Standard library imports
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, Any

if TYPE_CHECKING:
    import clang.cindex as cindex

# Optional Clang bindings - required for code analysis
try:
    import clang.cindex as cindex
except ImportError as exc:
    raise SystemExit(
        "libclang bindings are missing. Install with: pip install libclang"
    ) from exc

# Local module imports
from backend import config, utils
from backend.logger import get_file_logger


# Kernel-specific edge types
REL_CONTAINS = "CONTAINS"
REL_CALLS = "CALLS"
REL_READS = "READS"
REL_WRITES = "WRITES"
REL_DEFINES = "DEFINES"
REL_INCLUDES = "INCLUDES"

# Node type constants (from NodeExtractor)
NODE_TYPES = {
    "REPO": "REPO",
    "SUBSYSTEM": "SUBSYSTEM",
    "FILE": "FILE",
    "HEADER": "HEADER",
    "STRUCT": "STRUCT",
    "FUNCTION": "FUNCTION",
    "GLOBAL_VAR": "GLOBAL_VAR",
    "MACRO": "MACRO",
}


class GraphBuilder:
    """Build graph edges between kernel nodes with hierarchical structure."""

    def __init__(self, source_root: Path):
        self.source_root = Path(source_root)
        self._clang_index = cindex.Index.create()

        # Use a file logger specific to this module
        self.logger = get_file_logger("GraphBuilder")
        self.logger.info(f"GraphBuilder initialized with source_root: {source_root}")

    def build_hierarchical_edges(self, nodes: List[Dict], compile_db: List[Dict]) -> List[Dict]:
        """Build hierarchical edges between kernel nodes.

        Args:
            nodes: List of hierarchical nodes from NodeExtractor
            compile_db: Compilation database entries

        Returns:
            List of edge dictionaries with kernel-specific edge types
        """
        self.logger.info(f"Building hierarchical edges from {len(nodes)} nodes and {len(compile_db)} compile db entries...")

        edges: Set[Tuple[Tuple[str, Any], ...]] = set()

        # Build lookup maps
        nodes_by_id = {node["id"]: node for node in nodes}
        nodes_by_hash = {node["hash"]: node for node in nodes}

        # Build CONTAINS edges from parent-child relationships
        self._build_contains_edges(nodes, edges)

        # Build DEFINES edges (FILE/HEADER -> STRUCT/MACRO)
        self._build_defines_edges(nodes, edges)

        # Build CALLS, READS, WRITES edges from code analysis
        self._build_code_analysis_edges(nodes, compile_db, edges, nodes_by_hash)

        # Build INCLUDES edges from header dependencies
        self._build_includes_edges(nodes, compile_db, edges, nodes_by_hash)

        formatted_edges = self._format_edges(edges)
        self.logger.info(f"Built {len(formatted_edges)} hierarchical edges")
        return formatted_edges

    def _build_contains_edges(self, nodes: List[Dict], edges: Set[Tuple[Tuple[str, Any], ...]]) -> None:
        """Build CONTAINS edges from parent-child relationships."""
        for node in nodes:
            parent_id = node.get("parent_id")
            if parent_id is not None:
                self._add_edge(edges, parent_id, node["id"], REL_CONTAINS)

    def _build_defines_edges(self, nodes: List[Dict], edges: Set[Tuple[Tuple[str, Any], ...]]) -> None:
        """Build DEFINES edges (FILE/HEADER -> STRUCT/MACRO)."""
        # Create map of file nodes by file path
        file_nodes_by_path = {}
        for node in nodes:
            if node["type"] in [NODE_TYPES["FILE"], NODE_TYPES["HEADER"]]:
                file_nodes_by_path[node["file"]] = node["id"]

        # For each STRUCT or MACRO node, find its parent file and create DEFINES edge
        for node in nodes:
            if node["type"] in [NODE_TYPES["STRUCT"], NODE_TYPES["MACRO"]]:
                file_id = file_nodes_by_path.get(node["file"])
                if file_id is not None:
                    self._add_edge(edges, file_id, node["id"], REL_DEFINES)

    def _build_code_analysis_edges(
        self,
        nodes: List[Dict],
        compile_db: List[Dict],
        edges: Set[Tuple[Tuple[str, Any], ...]],
        nodes_by_hash: Dict[str, Dict]
    ) -> None:
        """Build CALLS, READS, WRITES edges from code analysis."""
        # Build lookup maps
        funcs_by_name = self._build_funcs_by_name(nodes)
        structs_by_name = self._build_structs_by_name(nodes)
        globals_by_name = self._build_globals_by_name(nodes)

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

            self.logger.debug(f"Analyzing code relationships in file: {rel_file}")

            # Process each function definition
            for cur in tu.cursor.get_children():
                if cur.kind != cindex.CursorKind.FUNCTION_DECL or not cur.is_definition():
                    continue
                if not cur.location.file:
                    continue
                if os.path.normpath(cur.location.file.name) != os.path.normpath(str(file_abs)):
                    continue

                func_name = cur.spelling
                func_id = self._find_node_id(func_name, NODE_TYPES["FUNCTION"], rel_file, nodes_by_hash)
                if func_id is None:
                    continue

                # Process function body for relationships
                self._process_function_body(
                    cur, func_id, rel_file, edges,
                    funcs_by_name, structs_by_name, globals_by_name,
                    nodes_by_hash
                )

    def _build_includes_edges(
        self,
        nodes: List[Dict],
        compile_db: List[Dict],
        edges: Set[Tuple[Tuple[str, Any], ...]],
        nodes_by_hash: Dict[str, Dict]
    ) -> None:
        """Build INCLUDES edges from header dependencies."""
        # Map header files to their node IDs
        header_nodes_by_path = {}
        for node in nodes:
            if node["type"] == NODE_TYPES["HEADER"]:
                header_nodes_by_path[node["file"]] = node["id"]

        for entry in compile_db:
            src_file = entry.get("file")
            if not src_file:
                continue

            file_abs, directory_abs = self._resolve_entry_file(entry)
            if file_abs is None:
                continue

            rel_file = utils.to_rel_xv6(str(file_abs))
            if rel_file is None:
                continue

            # Find the file node
            file_id = None
            for node in nodes:
                if node["type"] in [NODE_TYPES["FILE"], NODE_TYPES["HEADER"]] and node["file"] == rel_file:
                    file_id = node["id"]
                    break

            if file_id is None:
                continue

            # Parse file to find include directives
            try:
                with open(file_abs, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue

            # Find all #include directives
            include_pattern = r'^\s*#\s*include\s+["<]([^">]+)[">]'
            for line in content.split('\n'):
                match = re.match(include_pattern, line.strip())
                if match:
                    included_file = match.group(1)
                    # Try to find the header node
                    # First check if it's a relative path
                    header_path = self._resolve_header_path(included_file, rel_file)
                    if header_path in header_nodes_by_path:
                        header_id = header_nodes_by_path[header_path]
                        self._add_edge(edges, file_id, header_id, REL_INCLUDES)

    def _build_funcs_by_name(self, nodes: List[Dict]) -> Dict[str, List[int]]:
        """Build map of function names to node IDs."""
        funcs_by_name = {}
        for node in nodes:
            if node["type"] == NODE_TYPES["FUNCTION"]:
                name = node["name"]
                if name not in funcs_by_name:
                    funcs_by_name[name] = []
                funcs_by_name[name].append(node["id"])
        return funcs_by_name

    def _build_structs_by_name(self, nodes: List[Dict]) -> Dict[str, List[int]]:
        """Build map of struct names to node IDs."""
        structs_by_name = {}
        for node in nodes:
            if node["type"] == NODE_TYPES["STRUCT"]:
                name = node["name"]
                if name not in structs_by_name:
                    structs_by_name[name] = []
                structs_by_name[name].append(node["id"])
        return structs_by_name

    def _build_globals_by_name(self, nodes: List[Dict]) -> Dict[str, List[int]]:
        """Build map of global variable names to node IDs."""
        globals_by_name = {}
        for node in nodes:
            if node["type"] == NODE_TYPES["GLOBAL_VAR"]:
                name = node["name"]
                if name not in globals_by_name:
                    globals_by_name[name] = []
                globals_by_name[name].append(node["id"])
        return globals_by_name

    def _find_node_id(self, name: str, node_type: str, file_path: str, nodes_by_hash: Dict[str, Dict]) -> Optional[int]:
        """Find node ID by name, type, and file path."""
        # Create a search key without code content
        search_key = f"{node_type}:{name}:{file_path}:"
        for hash_key, node in nodes_by_hash.items():
            if hash_key.startswith(search_key):
                return node["id"]
        return None

    def _process_function_body(
        self,
        cursor,
        func_id: int,
        rel_file: str,
        edges: Set[Tuple[Tuple[str, Any], ...]],
        funcs_by_name: Dict[str, List[int]],
        structs_by_name: Dict[str, List[int]],
        globals_by_name: Dict[str, List[int]],
        nodes_by_hash: Dict[str, Dict]
    ) -> None:
        """Process a function body to extract CALLS, READS, WRITES edges."""
        # Helper function to check if a cursor is in a write context
        def _is_write_context(cursor) -> bool:
            parent = cursor.semantic_parent
            if not parent:
                return False

            # Check if cursor is LHS of assignment
            if parent.kind == cindex.CursorKind.BINARY_OPERATOR:
                tokens = list(parent.get_tokens())
                if len(tokens) >= 2 and tokens[1].spelling == "=":
                    children = list(parent.get_children())
                    if children and children[0] == cursor:
                        return True

            # Check if cursor is operand of increment/decrement operator
            if parent.kind == cindex.CursorKind.UNARY_OPERATOR:
                tokens = list(parent.get_tokens())
                if tokens and tokens[0].spelling in ("++", "--"):
                    return True

            return False

        # Helper function to check if a cursor is in a read context
        def _is_read_context(cursor) -> bool:
            parent = cursor.semantic_parent
            if not parent:
                return False

            # Check if cursor is RHS of assignment
            if parent.kind == cindex.CursorKind.BINARY_OPERATOR:
                tokens = list(parent.get_tokens())
                if len(tokens) >= 2 and tokens[1].spelling == "=":
                    children = list(parent.get_children())
                    if len(children) >= 2 and children[1] == cursor:
                        return True

            # Check if cursor is in conditional or loop statement
            current = parent
            while current:
                if current.kind in (
                    cindex.CursorKind.IF_STMT,
                    cindex.CursorKind.WHILE_STMT,
                    cindex.CursorKind.FOR_STMT,
                    cindex.CursorKind.DO_STMT,
                    cindex.CursorKind.SWITCH_STMT,
                ):
                    return True
                current = current.semantic_parent

            # Check if cursor is function argument
            if parent.kind == cindex.CursorKind.CALL_EXPR:
                return True

            return False

        # Helper function to get call context metadata
        def _get_call_context(cursor) -> Dict[str, bool]:
            metadata = {"is_interrupt": False}
            current = cursor.semantic_parent
            while current:
                # Check if this is a trap/interrupt handler
                func_name = cursor.spelling.lower() if cursor.spelling else ""
                if "trap" in func_name or "interrupt" in func_name or "handler" in func_name:
                    metadata["is_interrupt"] = True
                current = current.semantic_parent
            return metadata

        # Process all AST nodes within the function body
        for node in self._iter_descendants(cursor):
            # Process CALL_EXPR - Function calls
            if node.kind == cindex.CursorKind.CALL_EXPR:
                callee = node.referenced.spelling if node.referenced else node.spelling
                if callee and callee in funcs_by_name:
                    metadata = _get_call_context(node)
                    for dst_id in funcs_by_name[callee]:
                        self._add_edge(edges, func_id, dst_id, REL_CALLS, metadata)

            # Process DECL_REF_EXPR - Global variable usage
            elif node.kind == cindex.CursorKind.DECL_REF_EXPR:
                ref = node.referenced
                if ref and ref.kind == cindex.CursorKind.VAR_DECL:
                    parent = ref.semantic_parent
                    if parent and parent.kind == cindex.CursorKind.TRANSLATION_UNIT:
                        var_name = ref.spelling
                        if var_name in globals_by_name:
                            if _is_write_context(node):
                                for dst_id in globals_by_name[var_name]:
                                    self._add_edge(edges, func_id, dst_id, REL_WRITES)
                            elif _is_read_context(node):
                                for dst_id in globals_by_name[var_name]:
                                    self._add_edge(edges, func_id, dst_id, REL_READS)

            # Process MEMBER_REF_EXPR - Struct field access
            elif node.kind == cindex.CursorKind.MEMBER_REF_EXPR:
                # For struct field accesses, we need to determine the struct type
                # This would require type inference - for now, we'll handle at struct level
                pass

            # Process TYPE_REF - Struct usage
            elif node.kind == cindex.CursorKind.TYPE_REF:
                ref = node.referenced
                if ref and ref.kind == cindex.CursorKind.STRUCT_DECL and ref.spelling:
                    struct_name = ref.spelling
                    if struct_name in structs_by_name:
                        if _is_write_context(node):
                            for dst_id in structs_by_name[struct_name]:
                                self._add_edge(edges, func_id, dst_id, REL_WRITES)
                        elif _is_read_context(node):
                            for dst_id in structs_by_name[struct_name]:
                                self._add_edge(edges, func_id, dst_id, REL_READS)

    def _resolve_header_path(self, included_file: str, source_file: str) -> str:
        """Resolve header file path from #include directive."""
        # Simple resolution - just return the included file name
        # In a real implementation, this would handle relative paths, system paths, etc.
        return included_file

    def _resolve_entry_file(self, entry: Dict) -> Tuple[Optional[Path], Optional[Path]]:
        """Resolve relative paths from a compilation database entry into absolute system paths."""
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
        """Parse a C source file into a Clang TranslationUnit."""
        options = cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        try:
            self.logger.debug(f"Parsing {file_abs} with compile commands args")
            tu = self._clang_index.parse(path=file_abs, args=args, options=options)
            self.logger.debug(f"Successfully parsed {file_abs}")
            return tu
        except Exception as e:
            self.logger.debug(f"Failed to parse {file_abs} with compile commands: {e}, trying fallback args")
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
                self.logger.debug(f"Parsing {file_abs} with fallback args")
                tu = self._clang_index.parse(path=file_abs, args=fallback_args, options=options)
                self.logger.debug(f"Successfully parsed {file_abs} with fallback args")
                return tu
            except Exception as e2:
                self.logger.warning(f"Failed to parse {file_abs} even with fallback args: {e2}")
                return None

    def _iter_descendants(self, node) -> Iterable:
        """Recursively yield all descendant nodes of an AST node (depth-first traversal)."""
        for child in node.get_children():
            yield child
            yield from self._iter_descendants(child)

    def _format_edges(self, edges: Set[Tuple[Tuple[str, Any], ...]]) -> List[Dict]:
        """Format edges from internal tuple representation to dictionary list for JSON serialization."""
        formatted_edges = []
        for edge_tuple in edges:
            edge_dict = {}
            for key, value in edge_tuple:
                if key == "metadata" and isinstance(value, tuple):
                    # Convert metadata tuple back to dict
                    edge_dict[key] = dict(value)
                else:
                    edge_dict[key] = value
            formatted_edges.append(edge_dict)
        return sorted(formatted_edges, key=lambda x: (x["type"], x["from"], x["to"]))

    def _add_edge(self, edge_set: Set[Tuple[Tuple[str, Any], ...]], src: int, dst: int, rel: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an edge to the edge set with canonical representation."""
        if src == dst:
            return
        # Create a canonical representation for set storage
        edge_tuple = (("from", src), ("to", dst), ("type", rel))
        if metadata:
            # Sort metadata items for consistent ordering
            metadata_items = tuple(sorted(metadata.items()))
            edge_tuple = edge_tuple + (("metadata", metadata_items),)
        edge_set.add(edge_tuple)