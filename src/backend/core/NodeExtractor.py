# Standard library imports
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Any

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


class NodeExtractor:
    """Extract hierarchical kernel nodes from source code with kernel-specific types."""

    # Kernel-specific node types
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

    # Map file extensions to node types
    FILE_TYPE_MAP = {
        ".c": "FILE",
        ".S": "FILE",  # Assembly files
        ".h": "HEADER",
    }

    # Map directories to kernel subsystems
    SUBSYSTEM_MAP = {
        "kernel": "Memory",
        "user": "User",
        "fs": "File System",
        "proc": "Process",
        "trap": "Trap/Interrupt",
        "syscall": "System Call",
        "mmu": "Memory Management",
        "driver": "Device Driver",
        "net": "Network",
        "sync": "Synchronization",
    }

    def __init__(self, source_root: Path):
        self.source_root = Path(source_root)
        self._clang_index = cindex.Index.create()

        # Use a file logger specific to this module
        self.logger = get_file_logger("NodeExtractor")
        self.logger.info(f"NodeExtractor initialized with source_root: {source_root}")

    def extract_hierarchical_nodes(self, compile_db: List[Dict]) -> List[Dict]:
        """Extract hierarchical kernel nodes from compilation database.

        Returns a list of node dictionaries with hierarchical relationships.
        Each node has: id, type, name, file, subsystem, code, and parent_id.
        """
        self.logger.info(f"Extracting hierarchical nodes from {len(compile_db)} compile database entries...")

        # First pass: collect all files and their contents
        file_nodes = self._extract_file_nodes(compile_db)

        # Second pass: extract code elements from each file
        all_nodes = []
        next_id = 1

        # Add REPO node as root
        repo_node = {
            "id": next_id,
            "type": self.NODE_TYPES["REPO"],
            "name": "xv6-riscv",
            "file": "",
            "subsystem": "",
            "code": "",
            "parent_id": None,
            "hash": self._generate_node_hash("REPO", "xv6-riscv", "", "")
        }
        all_nodes.append(repo_node)
        repo_id = next_id
        next_id += 1

        # Group files by subsystem
        subsystem_nodes = {}
        for file_node in file_nodes:
            subsystem_name = self._determine_subsystem(file_node["file"])

            # Create subsystem node if not exists
            if subsystem_name not in subsystem_nodes:
                subsystem_node = {
                    "id": next_id,
                    "type": self.NODE_TYPES["SUBSYSTEM"],
                    "name": subsystem_name,
                    "file": "",
                    "subsystem": subsystem_name,
                    "code": "",
                    "parent_id": repo_id,
                    "hash": self._generate_node_hash("SUBSYSTEM", subsystem_name, "", "")
                }
                all_nodes.append(subsystem_node)
                subsystem_nodes[subsystem_name] = next_id
                next_id += 1

            # Update file node with parent_id
            file_node["parent_id"] = subsystem_nodes[subsystem_name]
            file_node["subsystem"] = subsystem_name
            all_nodes.append(file_node)

            # Extract code elements from this file
            code_nodes = self._extract_code_elements_from_file(
                file_node["file"], file_node["id"], subsystem_name, compile_db
            )
            for code_node in code_nodes:
                code_node["id"] = next_id
                next_id += 1
                all_nodes.append(code_node)

        self.logger.info(f"Extracted {len(all_nodes)} hierarchical nodes")
        return all_nodes

    def _extract_file_nodes(self, compile_db: List[Dict]) -> List[Dict]:
        """Extract FILE and HEADER nodes from compilation database."""
        file_nodes = []
        seen_files: Set[str] = set()
        next_id = 1

        for entry in compile_db:
            src_file = entry.get("file")
            if not src_file:
                continue

            # Determine file type based on extension
            file_path = Path(str(src_file))
            ext = file_path.suffix.lower()
            if ext not in self.FILE_TYPE_MAP:
                continue

            file_type = self.FILE_TYPE_MAP[ext]
            rel_file = utils.to_rel_xv6(str(src_file))
            if rel_file is None:
                continue

            # Skip if already processed
            if rel_file in seen_files:
                continue
            seen_files.add(rel_file)

            # Read file content
            file_abs, _ = self._resolve_entry_file(entry)
            if file_abs is None or not file_abs.exists():
                continue

            try:
                with open(file_abs, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except OSError:
                code = ""

            file_node = {
                "id": next_id,
                "type": file_type,
                "name": file_path.name,
                "file": rel_file,
                "subsystem": "",  # Will be set later
                "code": code,
                "parent_id": None,  # Will be set later
                "hash": self._generate_node_hash(file_type, file_path.name, rel_file, code)
            }
            file_nodes.append(file_node)
            next_id += 1

        return file_nodes

    def _extract_code_elements_from_file(
        self, rel_file: str, file_id: int, subsystem: str, compile_db: List[Dict]
    ) -> List[Dict]:
        """Extract code elements (STRUCT, FUNCTION, GLOBAL_VAR, MACRO) from a file."""
        # Find the compilation database entry for this file
        entry = None
        for db_entry in compile_db:
            db_file = db_entry.get("file", "")
            if utils.to_rel_xv6(str(db_file)) == rel_file:
                entry = db_entry
                break

        if entry is None:
            return []

        file_abs, directory_abs = self._resolve_entry_file(entry)
        if file_abs is None:
            return []

        args = utils.sanitize_compile_args(entry.get("arguments", []), str(entry.get("file", "")), str(directory_abs))
        if not args:
            return []

        tu = self._parse_tu(str(file_abs), args)
        if tu is None:
            return []

        code_nodes = []
        seen_elements: Set[Tuple[str, str, str]] = set()  # (type, name, code_hash)

        for node in self._iter_descendants(tu.cursor):
            elem_type = self._cursor_to_element_type(node)
            if elem_type is None:
                continue

            if not node.location.file:
                continue

            node_file = Path(os.path.normpath(node.location.file.name))
            if utils.to_rel_xv6(str(node_file)) != rel_file:
                continue

            # Read source lines for this node
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

            # Deduplicate
            dedup_key = (elem_type, name, code_hash)
            if dedup_key in seen_elements:
                continue
            seen_elements.add(dedup_key)

            # Map internal type to kernel node type
            kernel_type = self._map_to_kernel_type(elem_type)

            code_node = {
                "id": 0,  # Will be assigned later
                "type": kernel_type,
                "name": name,
                "file": rel_file,
                "subsystem": subsystem,
                "code": raw_code,
                "parent_id": file_id,
                "hash": self._generate_node_hash(kernel_type, name, rel_file, raw_code)
            }
            code_nodes.append(code_node)

        return code_nodes

    def _cursor_to_element_type(self, node) -> Optional[str]:
        """Determine the type of a given AST node."""
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

    def _map_to_kernel_type(self, internal_type: str) -> str:
        """Map internal type name to kernel node type."""
        type_map = {
            "function": self.NODE_TYPES["FUNCTION"],
            "struct": self.NODE_TYPES["STRUCT"],
            "global_var": self.NODE_TYPES["GLOBAL_VAR"],
            "macro": self.NODE_TYPES["MACRO"],
        }
        return type_map.get(internal_type, internal_type.upper())

    def _determine_subsystem(self, rel_file: str) -> str:
        """Determine kernel subsystem based on file path."""
        # Extract directory from file path
        file_path = Path(rel_file)
        if len(file_path.parts) > 1:
            directory = file_path.parts[0]
            # Check if directory maps to a known subsystem
            for dir_pattern, subsystem in self.SUBSYSTEM_MAP.items():
                if directory == dir_pattern or dir_pattern in directory:
                    return subsystem

        # Default to directory name or "Core"
        if len(file_path.parts) > 1:
            return file_path.parts[0].title()
        return "Core"

    def _generate_node_hash(self, node_type: str, name: str, file: str, code: str) -> str:
        """Generate deterministic MD5 hash for node identification."""
        content = f"{node_type}:{name}:{file}:{code}"
        return hashlib.md5(content.encode()).hexdigest()

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

    def _iter_descendants(self, node) -> List[Any]:
        """Recursively yield all descendant nodes of an AST node."""
        descendants = []
        for child in node.get_children():
            descendants.append(child)
            descendants.extend(self._iter_descendants(child))
        return descendants

    def _read_lines(self, file_path: Path) -> Optional[List[str]]:
        """Read the entire content of a file and return it as a list of lines."""
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
                return fh.read().splitlines()
        except OSError:
            return None

    def _source_range(self, node, source_lines: List[str]) -> str:
        """Extract the exact source code text for an AST node."""
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