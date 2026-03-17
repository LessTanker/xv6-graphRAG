import json
import logging
import re
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
from sentence_transformers import SentenceTransformer

try:
    from backend import config, utils
except ImportError:
    import config  # type: ignore
    import utils  # type: ignore


@dataclass
class RetrievalResult:
    seeds: List[Tuple[Dict[str, Any], float, float]]
    related_chunks: List[Dict[str, Any]]


class GraphRetriever:
    _HIGH_INDEGREE_THRESHOLD = 6

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.chunks = utils.load_metadata(config.CHUNKS_METADATA_PATH)
        self.edges = utils.load_edges(config.GRAPH_EDGES_PATH)
        self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        self.chunks_by_id = {chunk.get("id"): chunk for chunk in self.chunks if "id" in chunk}
        self.graph = self._build_graph_indices(self.edges)
        self.node_to_community = self._build_node_community_map()

    def retrieve(self, query_embedding, plan: Dict[str, Any], k: int = 10) -> RetrievalResult:
        seed_results = self._embedding_search(query_embedding=query_embedding, k=k)
        top_results = seed_results[:3]
        related_ids = self._apply_traversal_plan(plan=plan, top_embedding_results=seed_results)
        related_nodes = [self.chunks_by_id[nid] for nid in sorted(related_ids) if nid in self.chunks_by_id]

        top_results, related_nodes = self._enrich_and_refresh(
            top_results=top_results,
            related_nodes=related_nodes,
            query_embedding=query_embedding,
            plan=plan,
            k=k,
        )

        return RetrievalResult(seeds=top_results, related_chunks=related_nodes)

    def _enrich_and_refresh(self, top_results, related_nodes, query_embedding, plan: Dict[str, Any], k: int):
        all_selected_chunks = [chunk for chunk, _, _ in top_results] + related_nodes
        metadata_updated = False
        missing_summary_chunks = [chunk for chunk in all_selected_chunks if not chunk.get("summary")]

        if missing_summary_chunks:
            print(
                f"[summary] {len(missing_summary_chunks)} chunk(s) missing summary. "
                "Generating summaries with LLM..."
            )

        for idx, chunk in enumerate(missing_summary_chunks, start=1):
            print(
                f"[summary] ({idx}/{len(missing_summary_chunks)}) "
                f"chunk id={chunk.get('id')} name={chunk.get('name')}"
            )
            summary, keywords = self._get_llm_summary_for_chunk(chunk)
            if not summary:
                print(f"[summary] skip chunk id={chunk.get('id')} (empty LLM output)")
                continue
            chunk["summary"] = summary
            chunk["keywords"] = keywords
            metadata_updated = True

        if not metadata_updated:
            if missing_summary_chunks:
                print("[summary] No new summaries were saved.")
            return top_results, related_nodes

        print("[summary] Saving updated metadata to data/chunks_metadata.json")
        utils.save_json(config.CHUNKS_METADATA_PATH, self.chunks)

        print("[index] Recomputing embeddings for updated metadata...")
        all_texts = [utils.chunk_to_text(c) for c in self.chunks]
        all_embeddings = self.model.encode(
            all_texts,
            batch_size=max(1, config.EMBEDDING_BATCH_SIZE),
            convert_to_numpy=True,
        ).astype("float32")

        dim = all_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(all_embeddings)
        faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
        print("[index] FAISS index refreshed. Re-running retrieval with updated embeddings...")

        seed_results = self._embedding_search(query_embedding=query_embedding, k=k)
        top_results = seed_results[:3]
        related_ids = self._apply_traversal_plan(plan=plan, top_embedding_results=seed_results)
        related_nodes = [self.chunks_by_id[nid] for nid in sorted(related_ids) if nid in self.chunks_by_id]
        print("[summary] Summary enrichment complete.")
        return top_results, related_nodes

    def _get_llm_summary_for_chunk(self, chunk: Dict[str, Any]) -> Tuple[str, List[str]]:
        if not config.LLM_API_URL or not config.LLM_TOKEN or not config.LLM_MODEL:
            return "", []

        source_code = chunk.get("code", "")
        prompt = (
            "Analyze the following xv6 code to generate a concise summary and a few relevant keywords.\n"
            f"NAME: {chunk.get('name')}\n"
            f"TYPE: {chunk.get('type')}\n"
            f"FILE: {chunk.get('file')}\n"
            f"CODE:\n{source_code}\n\n"
            "Response ONLY in JSON format:\n"
            '{"summary": "...", "keywords": ["kw1", "kw2", ...]}'
        )

        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are an xv6 code parser. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }

        req = urllib.request.Request(
            config.LLM_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.LLM_TOKEN}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            content = data.get("choices", [])[0].get("message", {}).get("content", "")
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()

            try:
                result = json.loads(content)
                summary = str(result.get("summary", "")).strip()
                keywords = result.get("keywords", [])
                if not isinstance(keywords, list):
                    keywords = []
                keywords = [str(k).strip() for k in keywords if str(k).strip()]
                return summary, keywords
            except Exception:
                # Fallback: keep a concise plain-text summary if model output is not valid JSON.
                fallback_summary = " ".join(content.split())[:280].strip()
                return fallback_summary, []
        except Exception as exc:
            logging.error("Error generating summary for chunk %s: %s", chunk.get("id"), exc)
            return "", []

    def _build_graph_indices(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        outgoing = defaultdict(set)
        incoming = defaultdict(set)
        outgoing_by_type = defaultdict(lambda: defaultdict(set))
        incoming_by_type = defaultdict(lambda: defaultdict(set))
        undirected = defaultdict(set)
        calls_indegree = Counter()

        for edge in edges:
            src = edge.get("from")
            dst = edge.get("to")
            edge_type = edge.get("type")
            if src is None or dst is None or src == dst:
                continue

            outgoing[src].add(dst)
            incoming[dst].add(src)
            undirected[src].add(dst)
            undirected[dst].add(src)

            if isinstance(edge_type, str) and edge_type:
                outgoing_by_type[edge_type][src].add(dst)
                incoming_by_type[edge_type][dst].add(src)
                if edge_type == "CALLS":
                    calls_indegree[dst] += 1

        return {
            "outgoing": outgoing,
            "incoming": incoming,
            "outgoing_by_type": outgoing_by_type,
            "incoming_by_type": incoming_by_type,
            "undirected": undirected,
            "calls_indegree": calls_indegree,
        }

    def _embedding_search(self, query_embedding, k: int) -> List[Tuple[Dict[str, Any], float, float]]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, k=k)
        results: List[Tuple[Dict[str, Any], float, float]] = []
        for i, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            distance = float(distances[0][i])
            similarity = float(1 / (1 + distance))
            results.append((chunk, similarity, distance))
        return results

    def _bfs_expand(self, start_ids: List[int], adjacency: Dict[int, Set[int]], depth: int, node_filter=None) -> Set[int]:
        if node_filter is not None:
            start_ids = [nid for nid in start_ids if node_filter(nid)]
        visited = set(start_ids)
        queue = deque((nid, 0) for nid in start_ids)
        expanded = set()

        while queue:
            node_id, d = queue.popleft()
            if d >= depth:
                continue
            for nxt in adjacency.get(node_id, set()):
                if node_filter is not None and not node_filter(nxt):
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                expanded.add(nxt)
                queue.append((nxt, d + 1))
        return expanded

    def _shortest_path_between(self, a: int, b: int, adjacency: Dict[int, Set[int]], max_depth: int = 4, node_filter=None) -> List[int]:
        if node_filter is not None and (not node_filter(a) or not node_filter(b)):
            return []
        if a == b:
            return [a]
        queue = deque([(a, [a])])
        visited = {a}
        while queue:
            node_id, path = queue.popleft()
            if len(path) - 1 >= max_depth:
                continue
            for nxt in adjacency.get(node_id, set()):
                if node_filter is not None and not node_filter(nxt):
                    continue
                if nxt in visited:
                    continue
                new_path = path + [nxt]
                if nxt == b:
                    return new_path
                visited.add(nxt)
                queue.append((nxt, new_path))
        return []

    def _query_mentions_node(self, node_id: int, query_text: str) -> bool:
        if not query_text:
            return False
        chunk = self.chunks_by_id.get(node_id, {})
        name = str(chunk.get("name", "")).strip().lower()
        if not name:
            return False
        # Match full identifier-like names (e.g. panic, printf) to avoid broad substring hits.
        return re.search(rf"(?<![a-zA-Z0-9_]){re.escape(name)}(?![a-zA-Z0-9_])", query_text) is not None

    def _filter_common_nodes(self, node_ids: Set[int], calls_indegree: Dict[int, int], query_text: str) -> Set[int]:
        if not node_ids:
            return set()

        filtered: Set[int] = set()
        for nid in node_ids:
            if calls_indegree.get(nid, 0) < self._HIGH_INDEGREE_THRESHOLD:
                filtered.add(nid)
                continue
            if self._query_mentions_node(nid, query_text):
                filtered.add(nid)
        return filtered

    def _resolve_module_seed_files(self, top_embedding_results, target_entity: str) -> Set[str]:
        if target_entity:
            normalized = target_entity.strip().lower().replace("\\", "/")
            if normalized.endswith(".c") or normalized.endswith(".h"):
                return {normalized}

        file_counter = Counter()
        for chunk, _, _ in top_embedding_results:
            file_name = str(chunk.get("file", "")).strip().lower()
            if file_name:
                file_counter[file_name] += 1
        return {name for name, _ in file_counter.most_common(2)}

    def _build_node_community_map(self) -> Dict[int, Set[int]]:
        if not config.COMMUNITIES_PATH.exists():
            return {}

        try:
            data = utils.load_json_object(config.COMMUNITIES_PATH, "communities.json")
            mapping = data.get("community_id_to_node_ids", {})
            if not isinstance(mapping, dict):
                return {}

            node_to_community: Dict[int, Set[int]] = defaultdict(set)
            for community_id, members in mapping.items():
                if not isinstance(members, list):
                    continue
                if not str(community_id).isdigit():
                    continue
                cid = int(community_id)
                for node_id in members:
                    if isinstance(node_id, int):
                        node_to_community[node_id].add(cid)
            return dict(node_to_community)
        except Exception as exc:
            logging.warning("Failed to load communities for retriever scope control: %s", exc)
            return {}

    def _apply_traversal_plan(self, plan: Dict[str, Any], top_embedding_results) -> Set[int]:
        strategy = plan["traversal_strategy"]
        seed_ids = [c.get("id") for c, _, _ in top_embedding_results if c.get("id") is not None]
        seed_set = set(seed_ids)
        related_ids: Set[int] = set()
        query_type = str(plan.get("query_type", "UNKNOWN")).upper()

        restricted_raw = plan.get("restricted_community_id")
        if isinstance(restricted_raw, int):
            restricted_community_id = restricted_raw
        elif isinstance(restricted_raw, str) and restricted_raw.strip().isdigit():
            restricted_community_id = int(restricted_raw.strip())
        else:
            restricted_community_id = None

        def community_filter(node_id: int) -> bool:
            if restricted_community_id is None:
                return True
            return restricted_community_id in self.node_to_community.get(node_id, set())

        seed_ids = [nid for nid in seed_ids if community_filter(nid)]
        seed_set = set(seed_ids)

        outgoing_calls = self.graph["outgoing_by_type"].get("CALLS", {})
        incoming_struct = self.graph["incoming_by_type"].get("USES_STRUCT", {})
        outgoing_struct = self.graph["outgoing_by_type"].get("USES_STRUCT", {})
        undirected = self.graph["undirected"]
        calls_indegree = self.graph.get("calls_indegree", {})
        query_text = " ".join(
            part.strip().lower()
            for part in [
                str(plan.get("raw_query", "")),
                str(plan.get("target_entity", "")),
            ]
            if part and part.strip()
        )

        if not seed_ids:
            return related_ids

        if strategy == "embedding_only":
            return related_ids

        callee_match = re.match(r"^embedding \+ callees depth=(\d+)$", strategy)
        if callee_match:
            depth = max(1, int(callee_match.group(1)))
            if query_type == "FILE_OR_MODULE" and restricted_community_id is not None:
                depth += 1
            callee_ids = self._bfs_expand(seed_ids[:3], outgoing_calls, depth=depth, node_filter=community_filter)
            callee_ids = self._filter_common_nodes(callee_ids, calls_indegree, query_text)
            return callee_ids - seed_set

        bfs_match = re.match(r"^embedding \+ bfs depth=(\d+)$", strategy)
        if bfs_match:
            depth = max(1, int(bfs_match.group(1)))
            if query_type == "FILE_OR_MODULE" and restricted_community_id is not None:
                depth += 1
            bfs_ids = self._bfs_expand(seed_ids[:3], outgoing_calls, depth=depth, node_filter=community_filter)
            bfs_ids = self._filter_common_nodes(bfs_ids, calls_indegree, query_text)
            return bfs_ids - seed_set

        if strategy == "struct_reference_expansion":
            struct_ids = {nid for nid in seed_ids if self.chunks_by_id.get(nid, {}).get("type") == "struct"}

            if not struct_ids:
                for seed_id in seed_ids[:5]:
                    for candidate in outgoing_struct.get(seed_id, set()):
                        if self.chunks_by_id.get(candidate, {}).get("type") == "struct":
                            struct_ids.add(candidate)

            if restricted_community_id is not None:
                struct_ids = {sid for sid in struct_ids if community_filter(sid)}

            related_ids.update(struct_ids)
            referrers: Set[int] = set()
            for sid in struct_ids:
                referrers.update(incoming_struct.get(sid, set()))
            if restricted_community_id is not None:
                referrers = {rid for rid in referrers if community_filter(rid)}
            related_ids.update(referrers)

            for rid in list(referrers):
                if self.chunks_by_id.get(rid, {}).get("type") == "function":
                    callees = set(outgoing_calls.get(rid, set()))
                    if restricted_community_id is not None:
                        callees = {nid for nid in callees if community_filter(nid)}
                    related_ids.update(callees)

            related_ids = self._filter_common_nodes(related_ids, calls_indegree, query_text)
            return related_ids - seed_set

        if strategy == "multi_seed_connection_search":
            candidate_seeds = seed_ids[:5]
            for i in range(len(candidate_seeds)):
                for j in range(i + 1, len(candidate_seeds)):
                    path = self._shortest_path_between(
                        candidate_seeds[i],
                        candidate_seeds[j],
                        undirected,
                        max_depth=5 if query_type == "FILE_OR_MODULE" and restricted_community_id is not None else 4,
                        node_filter=community_filter,
                    )
                    if path:
                        related_ids.update(path)

            if not related_ids:
                related_ids = self._bfs_expand(candidate_seeds, undirected, depth=1, node_filter=community_filter)
            related_ids = self._filter_common_nodes(related_ids, calls_indegree, query_text)
            return related_ids - seed_set

        if strategy == "module_nodes_expansion":
            target_files = self._resolve_module_seed_files(top_embedding_results, plan.get("target_entity", ""))
            for node_id, chunk in self.chunks_by_id.items():
                file_name = str(chunk.get("file", "")).strip().lower()
                if file_name in target_files:
                    if restricted_community_id is not None and not community_filter(node_id):
                        continue
                    related_ids.add(node_id)

            depth = 2 if restricted_community_id is not None else 1
            module_depth_one = self._bfs_expand(list(related_ids), undirected, depth=depth, node_filter=community_filter)
            related_ids.update(module_depth_one)
            related_ids = self._filter_common_nodes(related_ids, calls_indegree, query_text)
            return related_ids - seed_set

        return related_ids
