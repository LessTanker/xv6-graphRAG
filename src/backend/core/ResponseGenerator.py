import json
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

try:
    from backend import config, utils
except ImportError:
    import config  # type: ignore
    import utils  # type: ignore


class ResponseGenerator:
    """Build final prompt from retrieval output and call LLM for final answer."""

    def __init__(self):
        self.chunks = utils.load_metadata(config.CHUNKS_METADATA_PATH)
        self.edges = utils.load_edges(config.GRAPH_EDGES_PATH)
        self.chunks_by_id = {
            chunk.get("id"): chunk
            for chunk in self.chunks
            if isinstance(chunk, dict) and chunk.get("id") is not None
        }
        self.adjacency = self._build_undirected_adj(self.edges)

    def generate(
        self,
        query: str,
        plan: Dict[str, Any],
        seeds,
        related_chunks,
        answer_language: str = config.LLM_RESPONSE_LANGUAGE,
    ) -> Dict[str, Any]:
        output = {
            "query": query,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query_plan": plan,
            "expert_path_match": plan.get("expert_path_match") if isinstance(plan.get("expert_path_match"), dict) else None,
            "top3_nodes": [
                {
                    "id": chunk.get("id"),
                    "type": chunk.get("type"),
                    "name": chunk.get("name"),
                    "file": chunk.get("file"),
                    "summary": chunk.get("summary"),
                    "keywords": chunk.get("keywords"),
                    "code": chunk.get("code"),
                    "similarity": similarity,
                    "distance": distance,
                }
                for chunk, similarity, distance in seeds
            ],
            "directly_related_nodes": [
                {
                    "id": chunk.get("id"),
                    "type": chunk.get("type"),
                    "name": chunk.get("name"),
                    "file": chunk.get("file"),
                    "summary": chunk.get("summary"),
                    "keywords": chunk.get("keywords"),
                    "code": chunk.get("code"),
                }
                for chunk in related_chunks
            ],
        }

        utils.save_json(config.SEARCH_RESULTS_PATH, output)
        prompt_markdown = self._build_prompt_markdown(output)
        config.PROMPT_PATH.write_text(prompt_markdown, encoding="utf-8")

        llm_answer, _raw = utils.call_llm(
            query,
            prompt_markdown,
            response_language=answer_language,
        )
        output["llm_response"] = llm_answer
        utils.save_json(config.SEARCH_RESULTS_PATH, output)
        return output

    def _build_undirected_adj(self, edges: List[dict]) -> Dict[int, Set[int]]:
        adj: Dict[int, Set[int]] = {}
        for edge in edges:
            a = edge.get("from")
            b = edge.get("to")
            if not isinstance(a, int) or not isinstance(b, int) or a == b:
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        return adj

    def _safe_text(self, value, default: str = "") -> str:
        if value is None:
            return default
        return str(value)

    def _render_code_block(self, code: str) -> List[str]:
        code = code.strip("\n")
        if not code:
            return ["```c", "// Code not found for this node id.", "```"]
        return ["```c", code, "```"]

    def _get_chunk_code_by_id(self, node_id: Any) -> str:
        if not isinstance(node_id, int):
            return ""
        chunk = self.chunks_by_id.get(node_id)
        if not isinstance(chunk, dict):
            return ""
        # Always source code from the canonical chunk store to avoid stale display fields.
        return self._safe_text(chunk.get("code"), "")

    def _build_prompt_markdown(self, query_result: Dict[str, Any]) -> str:
        top_nodes = query_result.get("top3_nodes", [])
        related_nodes = query_result.get("directly_related_nodes", [])
        expert_path_match = query_result.get("expert_path_match")
        related_by_id = {
            node.get("id"): node
            for node in related_nodes
            if isinstance(node, dict) and isinstance(node.get("id"), int)
        }

        lines: List[str] = []
        lines.append("# LLM Context")
        lines.append("")
        lines.append("## Query")
        lines.append(self._safe_text(query_result.get("query"), ""))
        lines.append("")
        lines.append("## Retrieval Metadata")
        lines.append(f"- Timestamp: `{self._safe_text(query_result.get('timestamp'), 'N/A')}`")
        lines.append(f"- Top center nodes: `{len(top_nodes)}`")
        lines.append(f"- Directly related nodes: `{len(related_by_id)}`")

        if isinstance(expert_path_match, dict):
            path_title = self._safe_text(expert_path_match.get("title"), "Unnamed expert path")
            path_similarity = expert_path_match.get("similarity")
            lines.append(f"- Expert path matched: `{path_title}`")
            if isinstance(path_similarity, (int, float)):
                lines.append(f"- Expert path similarity: `{path_similarity:.4f}`")

        lines.append("")
        if isinstance(expert_path_match, dict):
            lines.append("## Expert Path Context (Forced)")
            lines.append("")
            lines.append(self._safe_text(expert_path_match.get("description"), ""))
            lines.append("")

            node_ids = expert_path_match.get("node_ids", [])
            if isinstance(node_ids, list):
                for rank, node_id in enumerate(node_ids, 1):
                    if not isinstance(node_id, int):
                        continue
                    chunk = self.chunks_by_id.get(node_id, {})
                    lines.append(f"### Path Node {rank}")
                    lines.append(f"- Name: `{self._safe_text(chunk.get('name'), 'unknown')}`")
                    lines.append(f"- ID: `{node_id}`")
                    lines.append(f"- Type: `{self._safe_text(chunk.get('type'), 'N/A')}`")
                    lines.append(f"- File: `{self._safe_text(chunk.get('file'), 'N/A')}`")
                    lines.extend(self._render_code_block(self._safe_text(chunk.get("code"), "")))
                    lines.append("")

        lines.append("## Hierarchical Code Context")
        lines.append("")

        for center_rank, center in enumerate(top_nodes, 1):
            center_id = center.get("id")
            lines.append(f"## Root {center_rank}")
            lines.append(f"### Center Node: `{self._safe_text(center.get('name'), 'unknown')}`")
            lines.append(f"- ID: `{self._safe_text(center.get('id'), 'N/A')}`")
            lines.append(f"- Type: `{self._safe_text(center.get('type'), 'N/A')}`")
            lines.append(f"- File: `{self._safe_text(center.get('file'), 'N/A')}`")
            if "similarity" in center:
                lines.append(f"- Similarity: `{center['similarity']}`")
            if "distance" in center:
                lines.append(f"- Distance: `{center['distance']}`")

            center_code = self._get_chunk_code_by_id(center_id)
            lines.append("")
            lines.append("#### Core Source Code")
            lines.extend(self._render_code_block(center_code))
            lines.append("")

            neighbor_ids = []
            if isinstance(center_id, int):
                for neighbor_id in sorted(self.adjacency.get(center_id, set())):
                    if neighbor_id in related_by_id:
                        neighbor_ids.append(neighbor_id)

            lines.append("#### Neighbors")
            if not neighbor_ids:
                lines.append("- None")
                lines.append("")
                continue

            for neighbor_rank, neighbor_id in enumerate(neighbor_ids, 1):
                neighbor_meta = related_by_id[neighbor_id]
                lines.append("")
                lines.append(f"- Child {neighbor_rank}")
                lines.append(f"  - Name: `{self._safe_text(neighbor_meta.get('name'), 'unknown')}`")
                lines.append(f"  - ID: `{self._safe_text(neighbor_meta.get('id'), 'N/A')}`")
                lines.append(f"  - Type: `{self._safe_text(neighbor_meta.get('type'), 'N/A')}`")
                lines.append(f"  - File: `{self._safe_text(neighbor_meta.get('file'), 'N/A')}`")
                if neighbor_meta.get("summary"):
                    lines.append(f"  - Summary: {neighbor_meta['summary']}")
                neighbor_chunk = self.chunks_by_id.get(neighbor_id, {})
                neighbor_code = self._safe_text(neighbor_chunk.get("code"), "")
                lines.extend(self._render_code_block(neighbor_code))

            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
