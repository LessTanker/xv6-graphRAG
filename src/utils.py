import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from src import config
except ImportError:
    import config  # type: ignore


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


def load_json_object(path: Path, name: str = "JSON file") -> Dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return data


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    return load_json_array(path, "chunks_metadata.json")


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
    api_url = config.LLM_API_URL
    api_key = config.LLM_TOKEN
    model = config.LLM_MODEL

    if not api_url or not api_key or not model:
        return (
            "LLM call skipped: missing LLM_API_URL, LLM_TOKEN, or LLM_MODEL in .env.",
            None,
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert operating-systems assistant. Use the provided context to answer the user query. "
                    "If context is insufficient, explicitly state uncertainty. "
                    f"Respond in {response_language}."
                ),
            },
            {
                "role": "user",
                "content": f"User query:\n{query}\n\nContext (prompt.md):\n{context_markdown}",
            },
        ],
        "temperature": 0.2,
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            return "LLM call succeeded but no choices were returned.", data
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            return "LLM call succeeded but message content is empty.", data
        return content, data
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            pass
        return f"LLM call failed with HTTP {e.code}: {err_body}", None
    except Exception as e:
        return f"LLM call failed: {e}", None


def to_rel_xv6(abs_path: str) -> Optional[str]:
    norm = os.path.normpath(abs_path)
    marker = f"xv6-riscv{os.sep}"
    if marker not in norm:
        return None
    rel = norm.split(marker, 1)[1]
    if rel.startswith(("kernel/", "user/")):
        return rel
    return None


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
