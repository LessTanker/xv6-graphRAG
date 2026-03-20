# Standard library imports
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Optional, Set, Tuple

# Local module imports
from backend import config, utils
from backend.core.LLMClient import LLMClient
from backend.logger import get_file_logger


REL_CALLS = "CALLS"


class ExpertPathManager:
    """Build and persist expert paths from graph and chunk metadata."""

    # Configuration constants
    MIN_CHAIN_LEN = 3
    CATALOG_SIZE = 420
    UTILITY_NAMES = {
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

    def __init__(self, chunks_path: Path, edges_path: Path, output_path: Path):
        self.chunks_path = chunks_path
        self.edges_path = edges_path
        self.output_path = output_path

        # Initialize LLM client
        self.llm_client = LLMClient()

        # Initialize instance variables for intermediate data
        self.chunks: List[Dict] = []
        self.edges: List[Dict] = []
        self.function_chunks: List[Dict] = []
        self.call_edges: Set[Tuple[int, int]] = set()
        self.out_degree = Counter()
        self.in_degree = Counter()
        self.call_out: DefaultDict[int, List[int]] = defaultdict(list)
        self.call_in: DefaultDict[int, List[int]] = defaultdict(list)
        self.name_to_nodes: DefaultDict[str, List[Dict]] = defaultdict(list)
        self.by_id: Dict[int, Dict] = {}
        self.catalog_items: List[Dict[str, str]] = []
        self.llm_paths: List[Dict[str, object]] = []
        self.final_paths: List[Dict[str, object]] = []

        # Use a file logger specific to this module
        self.logger = get_file_logger("ExpertPathManager")
        self.logger.info(f"ExpertPathManager initialized with chunks_path={chunks_path}, edges_path={edges_path}, output_path={output_path}")

    # Load chunks and edges from disk and build basic data structures.
    # This method must be called before any other processing methods.
    def prepare_data(self) -> None:
        self.logger.info("Preparing data: loading chunks and edges...")
        self.chunks = utils.load_metadata(self.chunks_path)
        self.edges = utils.load_edges(self.edges_path)
        self.logger.info(f"Loaded {len(self.chunks)} chunks and {len(self.edges)} edges")

        self._build_function_graph()
        self.logger.info("Data preparation completed")

    # Build call graph structures from edges: call_edges, out_degree, in_degree, call_out, call_in.
    def _build_function_graph(self) -> None:
        self.logger.info("Building function call graph...")

        # Filter function chunks
        self.function_chunks = [
            chunk
            for chunk in self.chunks
            if isinstance(chunk, dict) and chunk.get("type") == "function" and isinstance(chunk.get("id"), int)
        ]

        if not self.function_chunks:
            self.logger.warning("No function chunks found")
            return

        self.logger.info(f"Found {len(self.function_chunks)} function chunks")

        # Reset graph structures
        self.call_edges.clear()
        self.out_degree.clear()
        self.in_degree.clear()
        self.call_out.clear()
        self.call_in.clear()

        # Build call graph
        for edge in self.edges:
            if str(edge.get("type", "")).strip() != REL_CALLS:
                continue
            src = edge.get("from")
            dst = edge.get("to")
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            self.call_edges.add((src, dst))
            self.out_degree[src] += 1
            self.in_degree[dst] += 1
            self.call_out[src].append(dst)
            self.call_in[dst].append(src)

        self.logger.info(f"Built call graph with {len(self.call_edges)} call edges")

    # Create a catalog of function metadata for LLM consumption.
    # Builds name_to_nodes and by_id maps, then creates catalog items with scores and connectivity info.
    def generate_catalog(self) -> None:
        if not self.function_chunks:
            self.logger.error("No function chunks available. Call prepare_data() first.")
            raise ValueError("No function chunks available. Call prepare_data() first.")

        self.logger.info("Generating function catalog for LLM...")

        # Build name and ID mappings
        self.name_to_nodes.clear()
        self.by_id.clear()
        for chunk in self.function_chunks:
            cid = int(chunk["id"])
            self.by_id[cid] = chunk
            self.name_to_nodes[str(chunk.get("name", "")).strip().lower()].append(chunk)

        self.logger.info(f"Built mappings: {len(self.by_id)} nodes by ID, {len(self.name_to_nodes)} unique names")

        # Create catalog items
        self.catalog_items = []
        for chunk in self.function_chunks:
            cid = int(chunk["id"])
            name = str(chunk.get("name", "")).strip()
            file_name = str(chunk.get("file", "")).strip()
            summary = str(chunk.get("summary", "")).strip()
            score = self.out_degree.get(cid, 0) + self.in_degree.get(cid, 0)
            top_callees = [str(self.by_id[nid].get("name", "")) for nid in self.call_out.get(cid, [])[:3] if nid in self.by_id]
            top_callers = [str(self.by_id[nid].get("name", "")) for nid in self.call_in.get(cid, [])[:3] if nid in self.by_id]

            self.catalog_items.append(
                {
                    "id": str(cid),
                    "name": name,
                    "file": file_name,
                    "summary": summary[:160],
                    "callees": ",".join(top_callees),
                    "callers": ",".join(top_callers),
                    "out": str(self.out_degree.get(cid, 0)),
                    "in": str(self.in_degree.get(cid, 0)),
                    "score": str(score),
                }
            )

        # Sort and limit catalog size
        self.catalog_items.sort(
            key=lambda item: (
                int(item["score"]),
                1 if str(item["file"]).startswith("kernel/") else 0,
                item["name"],
            ),
            reverse=True,
        )
        self.catalog_items = self.catalog_items[:self.CATALOG_SIZE]

        self.logger.info(f"Generated catalog with {len(self.catalog_items)} items")

    # Call LLM API to generate initial expert paths based on the function catalog.
    # Builds a prompt from catalog items and parses the LLM response.
    def call_llm_for_paths(self) -> None:
        if not self.catalog_items:
            self.logger.error("No catalog items available. Call generate_catalog() first.")
            raise ValueError("No catalog items available. Call generate_catalog() first.")

        self.logger.info("Calling LLM to generate expert paths...")

        max_count = max(1, int(config.EXPERT_PATH_MAX_COUNT))
        prompt_lines = self._build_llm_prompt(max_count)

        self.logger.info(f"Built LLM prompt with {len(prompt_lines)} lines")
        self.logger.debug(f"LLM prompt preview: {prompt_lines[:5]}...")

        raw = self.llm_client.call_api_simple(
            query="From the catalog, identify core xv6 call chains and function order.",
            context_markdown="\n".join(prompt_lines),
            response_language="English",
        )

        self.logger.info("LLM API call completed, parsing response...")
        parsed = self._parse_json_object(raw)
        self.llm_paths = parsed.get("paths", []) if isinstance(parsed, dict) else []
        if not isinstance(self.llm_paths, list):
            self.llm_paths = []

        self.logger.info(f"Parsed {len(self.llm_paths)} initial paths from LLM")

    # Build LLM prompt lines from catalog items.
    def _build_llm_prompt(self, max_count: int) -> List[str]:
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
        for idx, item in enumerate(self.catalog_items, 1):
            prompt_lines.append(
                f"{idx}. id={item['id']} {item['name']} ({item['file']}) score={item['score']} in={item['in']} out={item['out']} "
                f"callers=[{item['callers']}] callees=[{item['callers']}] summary={item['summary']}"
            )
        return prompt_lines

    # Process and validate LLM-generated paths, applying connectivity checks, theme detection, and deduplication.
    def process_paths(self) -> None:
        if not self.llm_paths:
            self.logger.error("No LLM paths available. Call call_llm_for_paths() first.")
            raise ValueError("No LLM paths available. Call call_llm_for_paths() first.")

        self.logger.info(f"Processing {len(self.llm_paths)} LLM-generated paths...")

        max_count = max(1, int(config.EXPERT_PATH_MAX_COUNT))
        self.final_paths = []
        seen: Set[Tuple[int, ...]] = set()

        for item in self.llm_paths:
            if len(self.final_paths) >= max_count:
                self.logger.info(f"Reached maximum path count ({max_count}), stopping processing")
                break

            if not isinstance(item, dict):
                self.logger.debug("Skipping non-dictionary path item")
                continue

            processed_path = self._process_single_path(item)
            if processed_path:
                key = tuple(processed_path["node_ids"])
                if key in seen:
                    self.logger.debug(f"Skipping duplicate path: {key}")
                    continue
                seen.add(key)
                self.final_paths.append(processed_path)
                self.logger.debug(f"Added path: {processed_path.get('title', 'Untitled')}")

        # Generate additional paths if needed
        if len(self.final_paths) < max_count:
            self._generate_additional_paths(max_count - len(self.final_paths), seen)

        self.logger.info(f"Processed {len(self.final_paths)} final expert paths")

    # Process a single path from LLM output, validating connectivity and extracting themes.
    def _process_single_path(self, item: Dict[str, object]) -> Optional[Dict[str, object]]:
        # Extract node IDs from the path item
        ids: List[int] = []
        raw_ids = item.get("node_ids", [])
        if isinstance(raw_ids, list):
            for nid in raw_ids:
                if not isinstance(nid, int):
                    continue
                if nid not in self.by_id:
                    continue
                if ids and ids[-1] == nid:
                    continue
                ids.append(nid)

        # Fallback to function names if node IDs insufficient
        if len(ids) < 4:
            funcs = item.get("functions", [])
            if isinstance(funcs, list):
                for name in funcs:
                    node = self._pick_node_by_name(str(name))
                    if node is None:
                        continue
                    node_id = int(node["id"])
                    if ids and ids[-1] == node_id:
                        continue
                    ids.append(node_id)

        if len(ids) < self.MIN_CHAIN_LEN:
            self.logger.debug(f"Path too short ({len(ids)} < {self.MIN_CHAIN_LEN}), skipping")
            return None

        # Validate and clean the path
        ids = self._ensure_connected(ids)
        ids = self._longest_connected_segment(ids)
        ids = self._trim_utility_tail(ids)

        if len(ids) < self.MIN_CHAIN_LEN:
            self.logger.debug(f"Path too short after cleaning ({len(ids)} < {self.MIN_CHAIN_LEN}), skipping")
            return None

        # Extract names and detect themes
        names = [str(self.by_id[nid].get("name", "")) for nid in ids if nid in self.by_id]
        themes = self._detect_themes(names)

        if not themes:
            self.logger.debug(f"No themes detected for path: {names}")
            return None

        return {
            "id": f"expert_{len(self.final_paths) + 1}",
            "title": str(item.get("title", "")).strip() or " -> ".join(names),
            "description": str(item.get("description", "")).strip(),
            "node_ids": ids,
            "node_names": names,
            "edge_types": [REL_CALLS] * (len(ids) - 1),
            "source": "llm_inferred",
            "themes": themes,
        }

    # Pick the most appropriate node by function name from name_to_nodes map.
    def _pick_node_by_name(self, name: str) -> Optional[Dict]:
        candidates = self.name_to_nodes.get(name.strip().lower(), [])
        if not candidates:
            self.logger.debug(f"No node found for name: {name}")
            return None
        candidates = sorted(
            candidates,
            key=lambda c: (
                1 if str(c.get("file", "")).startswith("kernel/") else 0,
                self.out_degree.get(int(c.get("id", -1)), 0),
            ),
            reverse=True,
        )
        return candidates[0]

    # Trim utility functions from the end of a path.
    def _trim_utility_tail(self, ids: List[int]) -> List[int]:
        trimmed = list(ids)
        while len(trimmed) > self.MIN_CHAIN_LEN:
            tail_name = str(self.by_id.get(trimmed[-1], {}).get("name", "")).strip().lower()
            if tail_name in self.UTILITY_NAMES:
                self.logger.debug(f"Trimming utility function from tail: {tail_name}")
                trimmed.pop()
                continue
            break
        return trimmed

    # Find a bridging node between two disconnected nodes if they share a common neighbor.
    def _bridge_one_hop(self, a: int, b: int) -> Optional[int]:
        for mid in self.call_out.get(a, []):
            if (mid, b) not in self.call_edges:
                continue
            mid_name = str(self.by_id.get(mid, {}).get("name", "")).strip().lower()
            if mid_name in self.UTILITY_NAMES:
                continue
            return mid
        return None

    # Ensure path connectivity by inserting bridging nodes where needed.
    def _ensure_connected(self, ids: List[int]) -> List[int]:
        if len(ids) < 2:
            return ids
        repaired = [ids[0]]
        for nxt in ids[1:]:
            prev = repaired[-1]
            if (prev, nxt) in self.call_edges:
                repaired.append(nxt)
                continue
            bridge = self._bridge_one_hop(prev, nxt)
            if bridge is not None and bridge not in repaired:
                self.logger.debug(f"Inserted bridge node {bridge} between {prev} and {nxt}")
                repaired.append(bridge)
                repaired.append(nxt)
                continue
            self.logger.debug(f"Could not bridge {prev} to {nxt}, stopping path")
            break
        return repaired

    # Extract the longest connected segment from a path.
    def _longest_connected_segment(self, ids: List[int]) -> List[int]:
        if len(ids) < 2:
            return ids
        best: List[int] = []
        cur: List[int] = [ids[0]]
        for nxt in ids[1:]:
            if (cur[-1], nxt) in self.call_edges:
                cur.append(nxt)
            else:
                if len(cur) > len(best):
                    best = cur[:]
                cur = [nxt]
        if len(cur) > len(best):
            best = cur[:]
        return best if best else ids

    # Detect themes from function names in a path.
    def _detect_themes(self, names: List[str]) -> List[str]:
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

    # Generate additional paths if initial LLM paths are insufficient.
    def _generate_additional_paths(self, needed: int, seen: Set[Tuple[int, ...]]) -> None:
        if needed <= 0:
            return

        self.logger.info(f"Generating {needed} additional paths...")
        existing = [" -> ".join(item.get("node_names", [])) for item in self.final_paths]

        extra_prompt = [
            "Generate additional unique xv6 core chains not overlapping with existing ones.",
            f"Need up to {needed} more chains.",
            "Return JSON only with key paths, same schema as before.",
            f"Existing chains to avoid: {existing}",
            "Function catalog:",
        ]
        for idx, item in enumerate(self.catalog_items, 1):
            extra_prompt.append(
                f"{idx}. id={item['id']} {item['name']} ({item['file']}) score={item['score']} in={item['in']} out={item['out']} "
                f"callers=[{item['callers']}] callees=[{item['callees']}] summary={item['summary']}"
            )

        self.logger.info("Calling LLM for additional paths...")
        extra_raw = self.llm_client.call_api_simple(
            query="Provide additional unique core xv6 chains.",
            context_markdown="\n".join(extra_prompt),
            response_language="English",
        )

        extra_parsed = self._parse_json_object(extra_raw)
        extra_paths = extra_parsed.get("paths", []) if isinstance(extra_parsed, dict) else []
        if not isinstance(extra_paths, list):
            extra_paths = []

        self.logger.info(f"Parsed {len(extra_paths)} additional paths from LLM")
        for item in extra_paths:
            if len(self.final_paths) >= needed + len(self.final_paths):
                break
            if not isinstance(item, dict):
                continue
            processed_path = self._process_single_path(item)
            if processed_path:
                key = tuple(processed_path["node_ids"])
                if key in seen:
                    continue
                seen.add(key)
                self.final_paths.append(processed_path)

        self.logger.info(f"Added {len(self.final_paths) - (len(self.final_paths) - needed)} additional paths")

    # Save processed expert paths to JSON file.
    def save_paths(self) -> None:
        if not self.final_paths:
            self.logger.error("No final paths available. Call process_paths() first.")
            raise ValueError("No final paths available. Call process_paths() first.")

        self.logger.info(f"Saving {len(self.final_paths)} expert paths to {self.output_path}")

        payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "path_count": len(self.final_paths),
            "expert_paths": self.final_paths,
        }

        utils.save_json(self.output_path, payload)
        self.logger.info("Expert paths saved successfully")


    # Parse JSON object from LLM response text, handling code blocks and malformed JSON.
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