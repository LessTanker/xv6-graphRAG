# Standard library imports
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Third-party library imports
import networkx as nx
import numpy as np

# Optional graph analysis libraries for community detection
try:
    import igraph as ig  # type: ignore
    import leidenalg  # type: ignore
except Exception:
    # Gracefully handle missing igraph or leidenalg
    ig = None
    leidenalg = None

# Local module imports with fallback for different execution contexts
try:
    # When running as part of the backend package
    from backend import config, utils
except ImportError:
    # When running as standalone script or from different context
    import config  # type: ignore
    import utils  # type: ignore


class CommunityManager:
    """Discover communities and build hierarchical summaries for module routing."""

    GLOBAL_COMMUNITY_NAME = "Global Kernel Utilities"
    MISC_COMMUNITY_NAME = "MISC"
    TARGET_COMMUNITY_MIN = config.COMMUNITY_TARGET_MIN
    TARGET_COMMUNITY_MAX = config.COMMUNITY_TARGET_MAX
    MIN_COMMUNITY_SIZE = config.COMMUNITY_MIN_SIZE
    GLOBAL_TOP_PERCENT = config.COMMUNITY_GLOBAL_TOP_PERCENT

    def __init__(
        self,
        edges_path: Path = config.GRAPH_EDGES_PATH,
        chunks_path: Path = config.CHUNKS_METADATA_PATH,
        output_path: Path = config.COMMUNITIES_PATH,
        prefer_static_partition: bool = False,
    ):
        self.edges_path = Path(edges_path)
        self.chunks_path = Path(chunks_path)
        self.output_path = Path(output_path)
        self.prefer_static_partition = prefer_static_partition

        self.chunks = utils.load_metadata(self.chunks_path)
        self.chunks_by_id = {
            chunk.get("id"): chunk
            for chunk in self.chunks
            if isinstance(chunk, dict) and isinstance(chunk.get("id"), int)
        }

        self.community_id_to_node_ids: Dict[int, List[int]] = {}
        self.community_summaries: Dict[int, str] = {}
        self.community_names: Dict[int, str] = {}
        self.community_records: List[Dict[str, Any]] = []
        self.node_memberships: Dict[int, Set[int]] = {}

        self.global_node_ids: List[int] = []
        self.global_summary: str = ""
        self.global_metadata: Dict[str, Any] = {}

        # Configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler
        log_dir = Path("log/backend")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "CommunityManager.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)

        self.logger.info(f"CommunityManager initialized with edges_path={edges_path}, chunks_path={chunks_path}, output_path={output_path}, prefer_static_partition={prefer_static_partition}")

    # This function executes the full community detection pipeline:
    # 1. Build directed and undirected graphs from edge data
    # 2. Identify global shared nodes (high in-degree utilities)
    # 3. Partition remaining normal nodes using Leiden algorithm or static path partitioning
    # 4. Post-process communities (merge small, coarsen to target size)
    # 5. Build community records and update internal state
    # Returns: Dict[int, List[int]] - Mapping from community ID to list of node IDs
    def run_leiden_algorithm(self) -> Dict[int, List[int]]:
        self.logger.info("Starting community detection pipeline")

        edges = utils.load_edges(self.edges_path)
        self.logger.info(f"Loaded edges from {self.edges_path}")

        undirected_graph = nx.Graph()
        directed_graph = nx.DiGraph()

        for node_id in self.chunks_by_id.keys():
            undirected_graph.add_node(node_id)
            directed_graph.add_node(node_id)

        edge_count = 0
        for edge in edges:
            src = edge.get("from")
            dst = edge.get("to")
            if not isinstance(src, int) or not isinstance(dst, int) or src == dst:
                continue

            if undirected_graph.has_edge(src, dst):
                undirected_graph[src][dst]["weight"] += 1.0
            else:
                undirected_graph.add_edge(src, dst, weight=1.0)

            if directed_graph.has_edge(src, dst):
                directed_graph[src][dst]["weight"] += 1.0
            else:
                directed_graph.add_edge(src, dst, weight=1.0)
            edge_count += 1

        self.logger.info(f"Graphs built with {undirected_graph.number_of_nodes()} nodes and {edge_count} edges")

        if undirected_graph.number_of_nodes() == 0:
            self.logger.warning("Empty graph detected, returning empty communities")
            self.community_id_to_node_ids = {}
            return self.community_id_to_node_ids

        self.logger.info("Identifying global shared nodes based on in-degree")
        global_nodes, indegree_threshold = self._identify_global_nodes(directed_graph)
        self.global_node_ids = global_nodes
        self.global_metadata = {
            "global_top_percent": self.GLOBAL_TOP_PERCENT,
            "in_degree_threshold": indegree_threshold,
            "global_node_count": len(global_nodes),
        }

        if self.global_node_ids:
            self.logger.info(
                f"Identified {len(self.global_node_ids)} GLOBAL_SHARED node(s) "
                f"(top {self.GLOBAL_TOP_PERCENT * 100:.1f}% by in-degree, threshold={indegree_threshold})"
            )
        else:
            self.logger.info("No global shared nodes identified")

        self.logger.info("Creating normal subgraph (excluding global nodes)")
        normal_node_ids = [nid for nid in undirected_graph.nodes() if nid not in set(self.global_node_ids)]
        normal_graph = undirected_graph.subgraph(normal_node_ids).copy()
        self.logger.info(f"Normal subgraph has {normal_graph.number_of_nodes()} nodes")

        if self.prefer_static_partition:
            self.logger.info("Using static path-based partitioning")
            partition_map = self._build_static_path_partitions(normal_node_ids)
        else:
            self.logger.info("Using Leiden algorithm for dynamic partitioning")
            partition_map = self._build_leiden_partitions(normal_graph)

        self.logger.info(f"Initial partitioning created {len(partition_map)} communities")

        self.logger.info(f"Merging small communities (min_size={self.MIN_COMMUNITY_SIZE})")
        partition_map = self._merge_small_communities(
            partition_map=partition_map,
            graph=normal_graph,
            min_size=self.MIN_COMMUNITY_SIZE,
        )

        self.logger.info(f"Coarsening communities to target max (target_max={self.TARGET_COMMUNITY_MAX})")
        partition_map = self._coarsen_communities(
            partition_map=partition_map,
            graph=normal_graph,
            target_max=self.TARGET_COMMUNITY_MAX,
        )

        final_communities = len(partition_map)
        self.logger.info(f"Post-processing complete: {final_communities} final communities")

        self.logger.info("Building community records")
        self.community_records = self._build_community_records(
            partition_map=partition_map,
            graph=undirected_graph,
            global_nodes=set(self.global_node_ids),
        )

        self.community_id_to_node_ids = {}
        self.community_names = {}
        self.node_memberships = defaultdict(set)
        for record in self.community_records:
            cid = int(record["community_id"])
            effective_nodes = [int(nid) for nid in record["effective_node_ids"]]
            self.community_id_to_node_ids[cid] = effective_nodes
            self.community_names[cid] = str(record["community_name"])
            for nid in effective_nodes:
                self.node_memberships[nid].add(cid)

        self.logger.info(
            f"Built {len(self.community_records)} merged community(ies) with {len(self.global_node_ids)} global shared nodes"
        )
        self.logger.info("Community detection pipeline completed successfully")

        return self.community_id_to_node_ids

    # Generate LLM summaries for each community and the global shared nodes.
    #    1. Ensures community detection has been run
    #    2. Summarizes global shared utility nodes (e.g., printf, panic)
    #    3. Iterates through each community, collecting evidence and calling LLM
    #    4. Stores summaries in both self.community_summaries and community_records
    # Returns:
    #    Dict[int, str]: Mapping from community ID to LLM-generated summary
    def summarize_communities(self) -> Dict[int, str]:
        if not self.community_id_to_node_ids:
            self.run_leiden_algorithm()

        self.global_summary = self._summarize_global_shared_nodes()

        summaries: Dict[int, str] = {}
        total = len(self.community_records)
        self.logger.info(f"Starting LLM summarization for {total} community(ies)...")

        for i, record in enumerate(self.community_records, start=1):
            cid = int(record["community_id"])
            cname = str(record["community_name"])
            core_nodes = [int(nid) for nid in record["core_node_ids"]]
            self.logger.info(
                f"({i}/{total}) Processing community_id={cid} name='{cname}' "
                f"core_nodes={len(core_nodes)} shared_globals={len(record['shared_global_node_ids'])}"
            )

            evidence_lines = self._collect_member_evidence(core_nodes, progress_prefix=f"({i}/{total})")
            files_text = ", ".join(record.get("core_files", [])[:8]) or "N/A"
            api_text = ", ".join(record.get("key_apis", [])[:12]) or "N/A"

            prompt = (
                "You are summarizing an xv6 subsystem module. "
                "Write 100-140 words in English as one compact paragraph. "
                "The paragraph must include: core functionality, core files, and key APIs."
            )
            context = (
                f"Community Name: {cname}\n"
                f"Core Files: {files_text}\n"
                f"Key APIs: {api_text}\n"
                f"Evidence:\n{chr(10).join(evidence_lines[:80])}\n"
                f"Global Utility Background:\n{self.global_summary}"
            )

            self.logger.info(f"({i}/{total}) Calling LLM for community_id={cid}...")
            content = utils.call_llm_api(
                query=prompt,
                context_markdown=context,
                response_language="English",
            ).strip()
            self.logger.info(f"({i}/{total}) LLM returned for community_id={cid}")

            if not content or content.startswith("LLM call failed") or content.startswith("LLM call skipped"):
                content = (
                    f"{cname} focuses on xv6 responsibilities around {files_text}. "
                    f"Representative APIs include {api_text}. "
                    "It collaborates with shared kernel utilities for diagnostics, synchronization, and reusable helpers."
                )
                self.logger.warning(f"({i}/{total}) Fallback summary used for community_id={cid}")

            summaries[cid] = content
            record["summary"] = content
            self.logger.info(f"({i}/{total}) Summary ready for community_id={cid}")

        self.community_summaries = summaries
        self.logger.info(f"Community summarization completed: {total} summaries generated")
        return summaries

    # Save all community structure, summaries, and metadata to a JSON file for downstream use.
    # Ensures community detection and summarization are complete, then serializes results to disk.
    def save_communities(self) -> None:
        if not self.community_id_to_node_ids:
            self.run_leiden_algorithm()
        if not self.community_summaries:
            self.summarize_communities()

        summaries_payload = {str(cid): self.community_summaries.get(cid, "") for cid in sorted(self.community_id_to_node_ids)}
        summaries_payload["GLOBAL_SHARED"] = self.global_summary

        payload = {
            "schema_version": "2.0",
            "partition_mode": "static_path" if self.prefer_static_partition else "leiden",
            "metadata": self.global_metadata,
            "Global_Nodes": {
                "community_name": self.GLOBAL_COMMUNITY_NAME,
                "node_ids": [int(nid) for nid in self.global_node_ids],
                "summary": self.global_summary,
            },
            "Community_Nodes": self.community_records,
            "node_memberships": {
                str(nid): sorted(int(cid) for cid in memberships)
                for nid, memberships in sorted(self.node_memberships.items(), key=lambda x: x[0])
            },
            # Backward-compatible fields for retriever modules.
            "community_id_to_node_ids": {
                str(cid): members for cid, members in sorted(self.community_id_to_node_ids.items())
            },
            "summaries": summaries_payload,
            "index_order": [int(cid) for cid in sorted(self.community_id_to_node_ids)],
            "community_names": {
                str(cid): self.community_names.get(cid, f"Community {cid}")
                for cid in sorted(self.community_id_to_node_ids)
            },
            "global_community_id": None,
        }
        utils.save_json(self.output_path, payload)

    # Identify top in-degree nodes as global utilities.
    def _identify_global_nodes(self, directed_graph: nx.DiGraph) -> Tuple[List[int], int]:
        if directed_graph.number_of_nodes() == 0:
            return [], 0

        indegree = {int(nid): int(deg) for nid, deg in directed_graph.in_degree()}
        if not indegree:
            return [], 0

        sorted_nodes = sorted(indegree.items(), key=lambda x: (-x[1], x[0]))
        top_k = max(1, int(np.ceil(len(sorted_nodes) * self.GLOBAL_TOP_PERCENT)))
        global_nodes = [nid for nid, _ in sorted_nodes[:top_k]]
        threshold = sorted_nodes[top_k - 1][1] if top_k - 1 < len(sorted_nodes) else sorted_nodes[-1][1]
        return sorted(global_nodes), int(threshold)

    # Partition nodes into communities based on file/module path structure.
    def _build_static_path_partitions(self, allowed_nodes: List[int]) -> Dict[int, List[int]]:
        groups = defaultdict(list)
        allowed = set(int(nid) for nid in allowed_nodes)
        for node_id, chunk in self.chunks_by_id.items():
            if int(node_id) not in allowed:
                continue
            file_name = str(chunk.get("file", "")).strip().replace("\\", "/")
            key = self._module_key_from_file(file_name)
            groups[key].append(node_id)

        community_map: Dict[int, List[int]] = {}
        for cid, key in enumerate(sorted(groups.keys())):
            community_map[cid] = sorted(groups[key])
        return community_map

    # Partition nodes into communities using the Leiden algorithm (or fallback to greedy modularity).
    def _build_leiden_partitions(self, graph: nx.Graph) -> Dict[int, List[int]]:
        if graph.number_of_nodes() == 0:
            return {}

        communities: List[List[int]] = []

        if ig is not None and leidenalg is not None:
            node_list = list(graph.nodes())
            node_to_idx = {nid: i for i, nid in enumerate(node_list)}
            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(node_list))

            weighted_edges = []
            weights = []
            for u, v, data in graph.edges(data=True):
                weighted_edges.append((node_to_idx[u], node_to_idx[v]))
                weights.append(float(data.get("weight", 1.0)))

            if weighted_edges:
                ig_graph.add_edges(weighted_edges)

            candidate_resolutions = [0.2, 0.35, 0.5, 0.7, 0.9, 1.1, 1.35, 1.7, 2.1]
            best_partition = None
            best_score = None
            for res in candidate_resolutions:
                partition = leidenalg.find_partition(
                    ig_graph,
                    leidenalg.RBConfigurationVertexPartition,
                    weights=weights if weights else None,
                    resolution_parameter=res,
                )
                count = len(partition)
                target_center = (self.TARGET_COMMUNITY_MIN + self.TARGET_COMMUNITY_MAX) / 2.0
                if self.TARGET_COMMUNITY_MIN <= count <= self.TARGET_COMMUNITY_MAX:
                    score = abs(count - target_center)
                else:
                    score = min(abs(count - self.TARGET_COMMUNITY_MIN), abs(count - self.TARGET_COMMUNITY_MAX)) + 10.0

                # Prefer better modularity if scores are similar.
                score += -float(partition.quality()) * 1e-6

                if best_score is None or score < best_score:
                    best_score = score
                    best_partition = partition

            if best_partition is not None:
                communities = [[node_list[idx] for idx in community] for community in best_partition]
        else:
            fallback = nx.algorithms.community.greedy_modularity_communities(graph)
            communities = [sorted(list(c)) for c in fallback]

        return {
            cid: sorted(int(nid) for nid in members)
            for cid, members in enumerate(communities)
            if members
        }

    # Merge communities smaller than min_size into larger neighbors or a misc group.
    def _merge_small_communities(self, partition_map: Dict[int, List[int]], graph: nx.Graph, min_size: int) -> Dict[int, List[int]]:
        communities = {int(cid): set(int(nid) for nid in members) for cid, members in partition_map.items() if members}
        if not communities:
            return {}

        misc_nodes: Set[int] = set()

        while True:
            valid = {cid: nodes for cid, nodes in communities.items() if nodes}
            small_ids = [cid for cid, nodes in valid.items() if len(nodes) < min_size]
            if not small_ids:
                break

            node_to_comm = {}
            for cid, nodes in valid.items():
                for nid in nodes:
                    node_to_comm[nid] = cid

            changed = False
            for cid in small_ids:
                nodes = communities.get(cid, set())
                if not nodes:
                    continue

                neighbor_weight = Counter()
                for nid in nodes:
                    for nbr in graph.neighbors(nid):
                        other = node_to_comm.get(int(nbr))
                        if other is None or other == cid:
                            continue
                        neighbor_weight[int(other)] += 1

                if neighbor_weight:
                    target_cid = neighbor_weight.most_common(1)[0][0]
                    communities[target_cid].update(nodes)
                    communities[cid] = set()
                    changed = True
                else:
                    misc_nodes.update(nodes)
                    communities[cid] = set()
                    changed = True

            if not changed:
                break

        merged = [sorted(nodes) for nodes in communities.values() if nodes]
        if misc_nodes:
            merged.append(sorted(misc_nodes))

        out: Dict[int, List[int]] = {}
        for cid, members in enumerate(merged):
            out[cid] = [int(nid) for nid in members]
        return out

    # Iteratively merge smallest communities until the total number does not exceed target_max.
    def _coarsen_communities(self, partition_map: Dict[int, List[int]], graph: nx.Graph, target_max: int) -> Dict[int, List[int]]:
        communities = {int(cid): set(int(nid) for nid in members) for cid, members in partition_map.items() if members}
        if len(communities) <= target_max:
            return partition_map

        while len([nodes for nodes in communities.values() if nodes]) > target_max:
            valid = {cid: nodes for cid, nodes in communities.items() if nodes}
            smallest_cid = min(valid.keys(), key=lambda c: len(valid[c]))
            smallest_nodes = valid[smallest_cid]

            node_to_comm = {}
            for cid, nodes in valid.items():
                for nid in nodes:
                    node_to_comm[nid] = cid

            neighbor_weight = Counter()
            for nid in smallest_nodes:
                for nbr in graph.neighbors(nid):
                    other = node_to_comm.get(int(nbr))
                    if other is None or other == smallest_cid:
                        continue
                    neighbor_weight[int(other)] += 1

            if neighbor_weight:
                target_cid = neighbor_weight.most_common(1)[0][0]
            else:
                target_cid = max((cid for cid in valid if cid != smallest_cid), key=lambda c: len(valid[c]))

            communities[target_cid].update(smallest_nodes)
            communities[smallest_cid] = set()

        out: Dict[int, List[int]] = {}
        for cid, nodes in enumerate([sorted(nodes) for nodes in communities.values() if nodes]):
            out[cid] = [int(nid) for nid in nodes]
        return out

    # Build structured records for each community, including core nodes, shared globals, files, and APIs.
    def _build_community_records(
        self,
        partition_map: Dict[int, List[int]],
        graph: nx.Graph,
        global_nodes: Set[int],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for cid, members in enumerate(members for _, members in sorted(partition_map.items())):
            core_set = set(int(nid) for nid in members)
            shared_globals = []
            for gid in sorted(global_nodes):
                connected = 0
                for nbr in graph.neighbors(gid):
                    if int(nbr) in core_set:
                        connected += 1
                if connected > 0:
                    shared_globals.append(int(gid))

            core_files, key_apis = self._extract_core_files_and_apis(list(core_set))
            is_misc = len(core_set) < self.MIN_COMMUNITY_SIZE
            community_name = self.MISC_COMMUNITY_NAME if is_misc else self._infer_community_name(core_files, key_apis, cid)
            effective_nodes = sorted(core_set.union(shared_globals))

            records.append(
                {
                    "community_id": int(cid),
                    "community_name": community_name,
                    "core_node_ids": sorted(core_set),
                    "shared_global_node_ids": shared_globals,
                    "effective_node_ids": effective_nodes,
                    "core_files": core_files,
                    "key_apis": key_apis,
                    "size": len(core_set),
                    "summary": "",
                }
            )
        return records
    
    # Generate an English summary for global shared utility nodes using LLM; fallback to template if LLM fails.
    def _summarize_global_shared_nodes(self) -> str:
        if not self.global_node_ids:
            self.global_summary = "No globally shared kernel utility nodes were detected."
            return self.global_summary

        self.logger.info(f"[community] Summarizing GLOBAL_SHARED set ({len(self.global_node_ids)} node(s))...")
        evidence = self._collect_member_evidence(self.global_node_ids, progress_prefix="(GLOBAL)")
        prompt = (
            "You are summarizing shared xv6 kernel utility functions. "
            "Write 100-140 words in English and explain cross-module utility value. "
            "Mention categories like panic/error handling, printf/logging, and synchronization helpers."
        )
        content = utils.call_llm_api(
            query=prompt,
            context_markdown="\n".join(evidence[:120]),
            response_language="English",
        ).strip()

        if not content or content.startswith("LLM call failed") or content.startswith("LLM call skipped"):
            content = (
                "Global Kernel Utilities contains heavily reused helper nodes such as diagnostics, panic handling, "
                "printing, locking, and other low-level support APIs. These nodes are cross-cutting dependencies "
                "shared by multiple subsystem communities and should always be mounted as global context during "
                "routing, retrieval, and reasoning."
            )
            self.logger.info("[community] (GLOBAL) Fallback summary used.")

        self.logger.info("[community] (GLOBAL) Summary ready.")
        self.global_summary = content
        return self.global_summary
    
    # Collect evidence text for a list of node IDs, reporting progress via logger.
    def _collect_member_evidence(self, node_ids: List[int], progress_prefix: str) -> List[str]:
        lines = []
        total = len(node_ids)
        for i, node_id in enumerate(node_ids, start=1):
            self.logger.info(f"[community] {progress_prefix} Building evidence {i}/{total}...")
            chunk = self.chunks_by_id.get(node_id, {})
            text = str(chunk.get("summary") or "").strip()
            if not text:
                name = str(chunk.get("name") or "")
                chunk_type = str(chunk.get("type") or "")
                file_name = str(chunk.get("file") or "")
                text = f"{chunk_type} {name} in {file_name}".strip()
            if text:
                lines.append(f"- node {node_id}: {text}")
        return lines

    # Extract the most frequent core files and API names from a list of node IDs.
    def _extract_core_files_and_apis(self, node_ids: List[int]) -> Tuple[List[str], List[str]]:
        file_counter = Counter()
        api_counter = Counter()

        for nid in node_ids:
            chunk = self.chunks_by_id.get(int(nid), {})
            file_name = str(chunk.get("file", "")).strip()
            name = str(chunk.get("name", "")).strip()
            typ = str(chunk.get("type", "")).strip()
            if file_name:
                file_counter[file_name] += 1
            if name and typ in {"function", "macro"}:
                api_counter[name] += 1

        return [k for k, _ in file_counter.most_common(8)], [k for k, _ in api_counter.most_common(12)]

    # Infer a human-readable community name based on core files and key APIs.
    def _infer_community_name(self, core_files: List[str], key_apis: List[str], cid: int) -> str:
        if core_files:
            head = core_files[0]
            if "/" in head:
                prefix = head.split("/", 1)[0]
                if prefix in {"kernel", "user"}:
                    return f"{prefix.capitalize()} Module {cid}"
        if key_apis:
            return f"API Cluster {key_apis[0]}"
        return f"Community {cid}"

    # Generate a module key from a file path for grouping nodes by module.
    def _module_key_from_file(self, file_name: str) -> str:
        if not file_name:
            return "unknown"

        parts = [part for part in file_name.split("/") if part]
        if len(parts) >= 3:
            return "/".join(parts[:2])

        if len(parts) == 2:
            top, leaf = parts
            stem = leaf.rsplit(".", 1)[0]
            return f"{top}/{stem}"

        stem = parts[0].rsplit(".", 1)[0]
        return stem
