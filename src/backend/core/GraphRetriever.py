import json
import logging
from backend.logger import get_file_logger
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

# Third-party library imports
import faiss
from sentence_transformers import SentenceTransformer

# Local module imports
from backend import config, utils
from backend.core.LLMClient import LLMClient


@dataclass
class RetrievalResult:
    """Container for graph retrieval results including seed nodes and related chunks."""
    seeds: List[Tuple[Dict[str, Any], float, float]]
    related_chunks: List[Dict[str, Any]]


class GraphRetriever:
    """Graph-based retrieval system for xv6 code knowledge graph."""

    _HIGH_INDEGREE_THRESHOLD = 6

    def __init__(self, model: SentenceTransformer):
        """Initialize GraphRetriever with embedding model and loaded graph data."""
        self.model = model


        # Use a file logger specific to this module
        self.logger = get_file_logger("GraphRetriever")
        self.logger.info("Initializing GraphRetriever with embedding model")

        # Initialize LLM client
        self.llm_client = LLMClient()

        # Load graph data and build indices
        self.chunks = utils.load_metadata(config.CHUNKS_METADATA_PATH)
        self.edges = utils.load_edges(config.GRAPH_EDGES_PATH)
        self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        self.chunks_by_id = {chunk.get("id"): chunk for chunk in self.chunks if "id" in chunk}
        self.graph = self._build_graph_indices(self.edges)
        self.node_to_community = self._build_node_community_map()

        self.logger.info(f"Loaded {len(self.chunks)} chunks, {len(self.edges)} edges")

    # Main entry point for graph-based retrieval.
    def retrieve(self, query_embedding, plan: Dict[str, Any], k: int = 10) -> RetrievalResult:
        """Retrieve relevant code chunks using graph traversal based on query plan.

        Args:
            query_embedding: Query vector for semantic search
            plan: Query plan containing traversal strategy and constraints
            k: Number of seed nodes to retrieve via embedding search

        Returns:
            RetrievalResult containing seed nodes and related chunks
        """
        self.logger.info("Starting graph retrieval with plan: %s", plan.get("traversal_strategy"))

        # Step 1: Get seed nodes via embedding search
        seed_results = self._embedding_search(query_embedding=query_embedding, k=k)
        top_results = seed_results[:3]

        # Step 2: Apply traversal plan to expand from seeds
        related_ids = self._apply_traversal_plan(plan=plan, top_embedding_results=seed_results)
        related_ids.update(self._get_forced_expert_path_node_ids(plan))

        # Step 3: Convert node IDs to chunk objects
        related_nodes = [self.chunks_by_id[nid] for nid in sorted(related_ids) if nid in self.chunks_by_id]

        # Step 4: Enrich metadata and refresh if needed
        top_results, related_nodes = self._enrich_and_refresh(
            top_results=top_results,
            related_nodes=related_nodes,
            query_embedding=query_embedding,
            plan=plan,
            k=k,
        )

        self.logger.info("Retrieval completed: %d seeds, %d related chunks",
                        len(top_results), len(related_nodes))
        return RetrievalResult(seeds=top_results, related_chunks=related_nodes)

    # Enrich metadata with LLM summaries and refresh indices if needed.
    def _enrich_and_refresh(self, top_results, related_nodes, query_embedding, plan: Dict[str, Any], k: int):
        # Identify chunks missing summaries
        all_selected_chunks = [chunk for chunk, _, _ in top_results] + related_nodes
        seen_ids = set()
        missing_summary_chunks = []

        for chunk in all_selected_chunks:
            chunk_id = chunk.get("id")
            if chunk_id is not None and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                if not chunk.get("summary"):
                    missing_summary_chunks.append(chunk)

        if not missing_summary_chunks:
            return top_results, related_nodes

        self.logger.info("Found %d unique chunk(s) missing summaries", len(missing_summary_chunks))

        # Generate summaries for missing chunks
        metadata_updated = False
        for idx, chunk in enumerate(missing_summary_chunks, start=1):
            self.logger.debug("(%d/%d) Generating summary for chunk id=%d name=%s",
                            idx, len(missing_summary_chunks), chunk.get('id'), chunk.get('name'))
            summary, keywords = self.llm_client.generate_summary_for_chunk(chunk)
            if not summary:
                self.logger.warning("Skipping chunk id=%d (empty LLM output)", chunk.get('id'))
                continue
            chunk["summary"] = summary
            chunk["keywords"] = keywords
            metadata_updated = True

        if not metadata_updated:
            self.logger.info("No new summaries were generated (LLM may be unavailable)")
            return top_results, related_nodes

        # Save updated metadata and refresh indices
        self.logger.info("Saving updated metadata to %s", config.CHUNKS_METADATA_PATH)
        utils.save_json(config.CHUNKS_METADATA_PATH, self.chunks)

        self.logger.info("Recomputing embeddings for updated metadata...")
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
        self.logger.info("FAISS index refreshed")

        # Re-run retrieval with updated embeddings
        self.logger.info("Re-running retrieval with updated embeddings...")
        seed_results = self._embedding_search(query_embedding=query_embedding, k=k)
        top_results = seed_results[:3]
        related_ids = self._apply_traversal_plan(plan=plan, top_embedding_results=seed_results)
        related_ids.update(self._get_forced_expert_path_node_ids(plan))
        related_nodes = [self.chunks_by_id[nid] for nid in sorted(related_ids) if nid in self.chunks_by_id]
        self.logger.info("Summary enrichment complete")
        return top_results, related_nodes

    # Get node IDs from expert path match in plan.
    def _get_forced_expert_path_node_ids(self, plan: Dict[str, Any]) -> Set[int]:
        match = plan.get("expert_path_match")
        if not isinstance(match, dict):
            return set()
        node_ids = match.get("node_ids", [])
        if not isinstance(node_ids, list):
            return set()
        return {nid for nid in node_ids if isinstance(nid, int)}

    # Get LLM-generated summary and keywords for a code chunk.
    # Build graph indices from edge list for efficient traversal.
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

    # Perform embedding search using FAISS index.
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

    # Perform BFS expansion from start nodes.
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

    def _filter_common_nodes(self, node_ids: Set[int], calls_indegree: Dict[int, int], query_text: str) -> Set[int]:
        if not node_ids:
            return set()

        filtered: Set[int] = set()
        for nid in node_ids:
            if calls_indegree.get(nid, 0) < self._HIGH_INDEGREE_THRESHOLD:
                filtered.add(nid)
                continue
            # Check if query mentions this node
            chunk = self.chunks_by_id.get(nid, {})
            name = str(chunk.get("name", "")).strip().lower()
            if name and query_text:
                if re.search(rf"(?<![a-zA-Z0-9_]){re.escape(name)}(?![a-zA-Z0-9_])", query_text):
                    filtered.add(nid)
        return filtered

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

    # Apply traversal plan to expand from seed nodes.
    def _apply_traversal_plan(self, plan: Dict[str, Any], top_embedding_results) -> Set[int]:
        strategy = plan["traversal_strategy"]
        self.logger.debug("Applying traversal strategy: %s", strategy)

        # Extract and prepare data for traversal
        context = self._prepare_traversal_context(plan, top_embedding_results)

        if not context["seed_ids"]:
            return set()

        # Apply strategy-specific traversal
        if strategy == "embedding_only":
            return set()

        # Handle different traversal strategies
        related_ids = self._apply_strategy(strategy, context)
        return related_ids - context["seed_set"]

    # Prepare context data for traversal operations.
    def _prepare_traversal_context(self, plan: Dict[str, Any], top_embedding_results) -> Dict[str, Any]:
        # Extract seed IDs from embedding results
        seed_ids = [c.get("id") for c, _, _ in top_embedding_results if c.get("id") is not None]
        seed_set = set(seed_ids)

        # Parse query type
        query_type = str(plan.get("query_type", "UNKNOWN")).upper()

        # Parse restricted community ID
        restricted_raw = plan.get("restricted_community_id")
        if isinstance(restricted_raw, int):
            restricted_community_id = restricted_raw
        elif isinstance(restricted_raw, str) and restricted_raw.strip().isdigit():
            restricted_community_id = int(restricted_raw.strip())
        else:
            restricted_community_id = None

        # Create community filter function
        def community_filter(node_id: int) -> bool:
            if restricted_community_id is None:
                return True
            return restricted_community_id in self.node_to_community.get(node_id, set())

        # Filter seeds by community
        seed_ids = [nid for nid in seed_ids if community_filter(nid)]
        seed_set = set(seed_ids)

        # Extract query text for filtering
        query_text = " ".join(
            part.strip().lower()
            for part in [
                str(plan.get("raw_query", "")),
                str(plan.get("target_entity", "")),
            ]
            if part and part.strip()
        )

        return {
            "plan": plan,
            "top_embedding_results": top_embedding_results,
            "strategy": plan["traversal_strategy"],
            "seed_ids": seed_ids,
            "seed_set": seed_set,
            "query_type": query_type,
            "restricted_community_id": restricted_community_id,
            "community_filter": community_filter,
            "query_text": query_text,
            "graph_data": {
                "outgoing_calls": self.graph["outgoing_by_type"].get("CALLS", {}),
                "incoming_struct": self.graph["incoming_by_type"].get("USES_STRUCT", {}),
                "outgoing_struct": self.graph["outgoing_by_type"].get("USES_STRUCT", {}),
                "undirected": self.graph["undirected"],
                "calls_indegree": self.graph.get("calls_indegree", {}),
            }
        }

    # Apply specific traversal strategy based on context.
    def _apply_strategy(self, strategy: str, context: Dict[str, Any]) -> Set[int]:
        # Pattern-based strategies
        callee_match = re.match(r"^embedding \+ (callees|bfs) depth=(\d+)$", strategy)
        if callee_match:
            depth = max(1, int(callee_match.group(2)))
            if context["query_type"] == "FILE_OR_MODULE" and context["restricted_community_id"] is not None:
                depth += 1

            callee_ids = self._bfs_expand(
                context["seed_ids"][:3],
                context["graph_data"]["outgoing_calls"],
                depth=depth,
                node_filter=context["community_filter"]
            )

            return self._filter_common_nodes(
                callee_ids,
                context["graph_data"]["calls_indegree"],
                context["query_text"]
            )

        # Named strategies
        strategy_handlers = {
            "struct_reference_expansion": self._apply_struct_reference_strategy,
            "multi_seed_connection_search": self._apply_multi_seed_strategy,
            "module_nodes_expansion": self._apply_module_expansion_strategy,
        }

        handler = strategy_handlers.get(strategy)
        if handler:
            return handler(context)

        return set()

    # Apply struct reference expansion strategy.
    def _apply_struct_reference_strategy(self, context: Dict[str, Any]) -> Set[int]:
        related_ids: Set[int] = set()

        # Find struct nodes from seeds
        struct_ids = {
            nid for nid in context["seed_ids"]
            if self.chunks_by_id.get(nid, {}).get("type") == "struct"
        }

        # If no structs in seeds, look for structs referenced by seeds
        if not struct_ids:
            for seed_id in context["seed_ids"][:5]:
                for candidate in context["graph_data"]["outgoing_struct"].get(seed_id, set()):
                    if self.chunks_by_id.get(candidate, {}).get("type") == "struct":
                        struct_ids.add(candidate)

        # Filter by community if restricted
        if context["restricted_community_id"] is not None:
            struct_ids = {
                sid for sid in struct_ids
                if context["community_filter"](sid)
            }

        # Add struct nodes and their referrers
        related_ids.update(struct_ids)

        referrers: Set[int] = set()
        for sid in struct_ids:
            referrers.update(context["graph_data"]["incoming_struct"].get(sid, set()))

        # Filter referrers by community if restricted
        if context["restricted_community_id"] is not None:
            referrers = {
                rid for rid in referrers
                if context["community_filter"](rid)
            }

        related_ids.update(referrers)

        # Expand from function referrers to their callees
        for rid in list(referrers):
            if self.chunks_by_id.get(rid, {}).get("type") == "function":
                callees = set(context["graph_data"]["outgoing_calls"].get(rid, set()))
                if context["restricted_community_id"] is not None:
                    callees = {
                        nid for nid in callees
                        if context["community_filter"](nid)
                    }
                related_ids.update(callees)

        # Filter common nodes
        return self._filter_common_nodes(
            related_ids,
            context["graph_data"]["calls_indegree"],
            context["query_text"]
        )
    
    def _apply_struct_reference_strategy(self, context: Dict[str, Any]) -> Set[int]:
        related_ids: Set[int] = set()

        # Find struct nodes from seeds
        struct_ids = {
            nid for nid in context["seed_ids"]
            if self.chunks_by_id.get(nid, {}).get("type") == "struct"
        }

        # If no structs in seeds, look for structs referenced by seeds
        if not struct_ids:
            for seed_id in context["seed_ids"][:5]:
                for candidate in context["graph_data"]["outgoing_struct"].get(seed_id, set()):
                    if self.chunks_by_id.get(candidate, {}).get("type") == "struct":
                        struct_ids.add(candidate)

        # Filter by community if restricted
        if context["restricted_community_id"] is not None:
            struct_ids = {
                sid for sid in struct_ids
                if context["community_filter"](sid)
            }

        # Add struct nodes and their referrers
        related_ids.update(struct_ids)

        referrers: Set[int] = set()
        for sid in struct_ids:
            referrers.update(context["graph_data"]["incoming_struct"].get(sid, set()))

        # Filter referrers by community if restricted
        if context["restricted_community_id"] is not None:
            referrers = {
                rid for rid in referrers
                if context["community_filter"](rid)
            }

        related_ids.update(referrers)

        # Expand from function referrers to their callees
        for rid in list(referrers):
            if self.chunks_by_id.get(rid, {}).get("type") == "function":
                callees = set(context["graph_data"]["outgoing_calls"].get(rid, set()))
                if context["restricted_community_id"] is not None:
                    callees = {
                        nid for nid in callees
                        if context["community_filter"](nid)
                    }
                related_ids.update(callees)

        # Filter common nodes
        return self._filter_common_nodes(
            related_ids,
            context["graph_data"]["calls_indegree"],
            context["query_text"]
        )

    # Apply multi-seed connection search strategy.
    def _apply_multi_seed_strategy(self, context: Dict[str, Any]) -> Set[int]:
        related_ids: Set[int] = set()
        candidate_seeds = context["seed_ids"][:5]

        # Find shortest paths between seed pairs
        for i in range(len(candidate_seeds)):
            for j in range(i + 1, len(candidate_seeds)):
                path = self._shortest_path_between(
                    candidate_seeds[i],
                    candidate_seeds[j],
                    context["graph_data"]["undirected"],
                    max_depth=5 if context["query_type"] == "FILE_OR_MODULE" and context["restricted_community_id"] is not None else 4,
                    node_filter=context["community_filter"],
                )
                if path:
                    related_ids.update(path)

        # Fallback to BFS if no paths found
        if not related_ids:
            related_ids = self._bfs_expand(
                candidate_seeds,
                context["graph_data"]["undirected"],
                depth=1,
                node_filter=context["community_filter"]
            )

        return self._filter_common_nodes(
            related_ids,
            context["graph_data"]["calls_indegree"],
            context["query_text"]
        )

    # Apply module nodes expansion strategy.
    def _apply_module_expansion_strategy(self, context: Dict[str, Any]) -> Set[int]:
        related_ids: Set[int] = set()

        # Get target files from embedding results
        target_entity = context.get("plan", {}).get("target_entity", "")
        target_files = set()

        if target_entity:
            normalized = target_entity.strip().lower().replace("\\", "/")
            if normalized.endswith(".c") or normalized.endswith(".h"):
                target_files.add(normalized)

        if not target_files:
            file_counter = Counter()
            for chunk, _, _ in context.get("top_embedding_results", []):
                file_name = str(chunk.get("file", "")).strip().lower()
                if file_name:
                    file_counter[file_name] += 1
            target_files = {name for name, _ in file_counter.most_common(2)}

        # Collect nodes from target files
        for node_id, chunk in self.chunks_by_id.items():
            file_name = str(chunk.get("file", "")).strip().lower()
            if file_name in target_files:
                if context["restricted_community_id"] is not None and not context["community_filter"](node_id):
                    continue
                related_ids.add(node_id)

        # Expand within module
        depth = 2 if context["restricted_community_id"] is not None else 1
        module_depth_one = self._bfs_expand(
            list(related_ids),
            context["graph_data"]["undirected"],
            depth=depth,
            node_filter=context["community_filter"]
        )
        related_ids.update(module_depth_one)

        return self._filter_common_nodes(
            related_ids,
            context["graph_data"]["calls_indegree"],
            context["query_text"]
        )

