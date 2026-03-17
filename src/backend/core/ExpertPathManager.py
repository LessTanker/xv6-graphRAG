import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

try:
    from backend import config, utils
except ImportError:
    import config  # type: ignore
    import utils  # type: ignore


REL_CALLS = "CALLS"


class ExpertPathManager:
    """Build and persist expert paths from graph and chunk metadata."""

    def __init__(self, chunks_path: Path, edges_path: Path, output_path: Path):
        self.chunks_path = chunks_path
        self.edges_path = edges_path
        self.output_path = output_path

    def build_and_save(self) -> None:
        chunks = utils.load_metadata(self.chunks_path)
        edges = utils.load_edges(self.edges_path)
        selected = self._extract_with_llm(chunks, edges)

        payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "path_count": len(selected),
            "expert_paths": selected,
        }
        utils.save_json(self.output_path, payload)

    def _extract_with_llm(self, chunks: List[Dict], edges: List[Dict]) -> List[Dict[str, object]]:
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
