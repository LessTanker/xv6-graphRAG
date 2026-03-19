# Standard library imports
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

# Local module imports
from backend import config


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_array(path: Path, name: str = "JSON file") -> List[Any]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{name} must contain a JSON array")
    return data


# Load and validate a JSON object from file.
def load_json_object(path: Path, name: str = "JSON file") -> Dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return data


# Load compile_commands.json and supplement with user-mode entries if missing.
def load_compile_db(path: Path, source_root: Path) -> List[Dict[str, Any]]:
    # Attempt to find xv6-riscv root more flexibly if direct path fails
    xv6_root = source_root / "xv6-riscv"
    if not xv6_root.exists():
        # Heuristic: search for kernel/defs.h to identify xv6 root
        potential_roots = list(source_root.glob("**/kernel/defs.h"))
        if potential_roots:
            xv6_root = potential_roots[0].parent.parent
        else:
            print(f"Warning: Could not locate xv6-riscv source root under {source_root}")
            return []

    if not path.exists():
        print(f"Warning: {path} not found. Building skeleton compile db from disk scan.")
        compile_db = []
    else:
        compile_db = load_json_array(path, "compile_commands.json")

    user_dir = xv6_root / "user"
    if not user_dir.exists():
        return compile_db

    augmented = list(compile_db)
    existing_user_files: Set[str] = set()

    for entry in compile_db:
        src_file = str(entry.get("file", "")).strip()
        if not src_file.endswith(".c"):
            continue
        rel_file = to_rel_xv6(src_file)
        if rel_file and rel_file.startswith("user/"):
            existing_user_files.add(rel_file)

    supplement_count = 0
    # Supplement missing user-mode entries. Makefile-generated compile_commands.json 
    # often omits standalone user programs, which are critical for cross-mode 
    # call-graph analysis (e.g., tracing syscalls from sh.c to sysfile.c).
    for c_file in sorted(user_dir.glob("*.c")):
        rel_file = f"user/{c_file.name}"
        if rel_file in existing_user_files:
            continue

        augmented.append(
            {
                "directory": str(xv6_root),
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
        supplement_count += 1
    
    if supplement_count > 0:
        print(f"Info: Supplemented {supplement_count} missing user-mode entries in compile database.")
        
    return augmented



# Load the code chunk metadata from a JSON file and return as a list of dictionaries.
def load_metadata(path: Path) -> List[Dict[str, Any]]:
    return load_json_array(path, "chunks_metadata.json")


# Load edge list from a graph_edges.json file.
def load_edges(path: Path) -> List[Dict[str, Any]]:
    data = load_json_object(path, "graph_edges.json")
    edges = data.get("edges", [])
    return edges if isinstance(edges, list) else []


def chunk_to_text(chunk: Dict[str, Any]) -> str:
    kws = ", ".join(chunk.get("keywords", []))
    code = chunk.get("code") or ""
    code = str(code)
    return (
        f"Kernel {chunk['type']} in xv6\n\n"
        f"Name: {chunk['name']}\n"
        f"File: {chunk['file']}\n\n"
        f"Summary:\n{chunk.get('summary', '')}\n\n"
        f"Keywords:\n{kws}\n\n"
        f"Code:\n{code}"
    )


def call_llm(
    query: str,
    context_markdown: str,
    response_language: str = "English",
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """DEPRECATED: Use LLMClient.call_with_context() instead.

    This function is kept for backward compatibility.
    """
    import warnings
    warnings.warn(
        "call_llm() is deprecated. Use LLMClient.call_with_context() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Create LLM client and call the new method
    from backend import LLMClient
    llm_client = LLMClient()
    content = llm_client.call_with_context(
        query=query,
        context_markdown=context_markdown,
        response_language=response_language,
    )

    # Return content and None for raw_data (maintaining backward compatibility)
    return content, None



def call_llm_api(
    query: str,
    context_markdown: str,
    response_language: str = "English",
) -> str:
    """DEPRECATED: Use LLMClient.call_api_simple() instead.

    This function is kept for backward compatibility.
    """
    import warnings
    warnings.warn(
        "call_llm_api() is deprecated. Use LLMClient.call_api_simple() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Create LLM client and call the new method
    from backend import LLMClient
    llm_client = LLMClient()
    return llm_client.call_api_simple(
        query=query,
        context_markdown=context_markdown,
        response_language=response_language,
    )


# Convert an absolute path under the xv6-riscv source tree to a relative path, keeping only files under kernel/ or user/.
# Used to unify indexing and display format, hiding absolute path differences across environments.
def to_rel_xv6(abs_path: str) -> Optional[str]:
    norm = os.path.normpath(abs_path)
    marker = f"xv6-riscv{os.sep}"
    if marker not in norm:
        return None
    rel = norm.split(marker, 1)[1]
    if rel.startswith(("kernel/", "user/")):
        return rel
    return None


# Clean up and reorganize compiler arguments from compile_commands.json to make
# them suitable for LibClang static analysis (e.g., removing output files and 
# dependency generation flags, while ensuring proper working directory).
def sanitize_compile_args(args: List[str], src_file: str, directory_abs: str) -> List[str]:
    if not args:
        return []

    out: List[str] = ["-working-directory", directory_abs]
    skip_next = False
    for i, a in enumerate(args):
        if i == 0:
            continue
        if skip_next:
            skip_next = False
            continue
        if a in {"-c", "-o", "-MMD", "-MF", "-MT", "-MQ"}:
            if a != "-c":
                skip_next = True
            continue
        if a.startswith("-o") and len(a) > 2:
            continue
        if a == src_file:
            continue
        out.append(a)

    if "-x" not in out:
        out.extend(["-x", "c"])
    return out
