# Standard library imports
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

if TYPE_CHECKING:
    import clang.cindex as cindex

# Third-party library imports
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional Clang bindings - required for code analysis
try:
    import clang.cindex as cindex
except ImportError as exc:
    raise SystemExit(
        "libclang bindings are missing. Install with: pip install libclang"
    ) from exc

# Local module imports
from backend import config, utils
from backend.core.CommunityManager import CommunityManager
from backend.core.ExpertPathManager import ExpertPathManager
from backend.core.GraphBuilder import GraphBuilder
from backend.core.NodeExtractor import NodeExtractor
from backend.core.LLMClient import LLMClient
from backend.logger import get_file_logger




class KnowledgeIndexer:
    """Offline pipeline: parse code, build graph, and create embeddings/index."""

    def __init__(self, source_root: Path, model: Optional[SentenceTransformer] = None):
        self.source_root = Path(source_root)
        self.model = model or SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self._clang_index = cindex.Index.create()

        # Use a file logger specific to this module
        self.logger = get_file_logger("KnowledgeIndexer")
        self.logger.info(f"KnowledgeIndexer initialized with source_root: {source_root}")

    # Check if all required indexing and analysis artifacts exist on disk.
    # If any are missing or force_rebuild is True, trigger the necessary
    # build or analysis pipelines.
    def ensure_ready(self, force_rebuild: bool = False) -> None:
        self.logger.info("Checking if indexing artifacts are ready...")
        required_paths = [
            config.CHUNKS_METADATA_PATH,
            config.GRAPH_EDGES_PATH,
            config.FAISS_INDEX_PATH,
            config.COMMUNITIES_PATH,
            config.COMMUNITY_FAISS_INDEX_PATH,
            config.EXPERT_PATHS_PATH,
        ]
        if force_rebuild or any(not p.exists() for p in required_paths):
            if force_rebuild:
                self.logger.info("Force rebuild requested, rebuilding all artifacts...")
            else:
                missing_paths = [p for p in required_paths if not p.exists()]
                self.logger.warning(f"Missing {len(missing_paths)} artifact(s): {missing_paths}, rebuilding...")
            self.build_all()
        else:
            self.logger.info("All indexing artifacts are ready.")

    # Orchestrate the complete offline indexing and analysis pipeline.
    def build_all(self) -> None:
        self.logger.info("Starting complete indexing pipeline...")
        compile_db = utils.load_compile_db(config.COMPILE_COMMANDS_PATH, self.source_root)

        # Use NodeExtractor to extract hierarchical nodes
        node_extractor = NodeExtractor(self.source_root)
        nodes = node_extractor.extract_hierarchical_nodes(compile_db)

        # Use GraphBuilder to build hierarchical edges
        graph_builder = GraphBuilder(self.source_root)
        edges = graph_builder.build_hierarchical_edges(nodes, compile_db)

        # Save nodes and edges
        utils.save_json(config.CHUNKS_METADATA_PATH, nodes)
        utils.save_json(config.GRAPH_EDGES_PATH, {"edges": edges})

        self._build_communities()
        self._build_expert_paths()

        self._build_and_save_chunks_embeddings()
        self._build_and_save_community_embeddings()
        self.logger.info(f"Indexing pipeline completed: {len(nodes)} nodes, {len(edges)} edges")

    # Run community detection and generate summaries using LLM.
    def _build_communities(self) -> None:
        self.logger.info("Starting community detection...")
        manager = CommunityManager(
            edges_path=config.GRAPH_EDGES_PATH,
            chunks_path=config.CHUNKS_METADATA_PATH,
            output_path=config.COMMUNITIES_PATH,
            prefer_static_partition=config.COMMUNITY_USE_STATIC_PATH_PARTITION,
        )
        manager.run_leiden_algorithm()
        manager.summarize_communities()
        manager.save_communities()
        self.logger.info("Community detection completed")

    # Extract and analyze core kernel call chains using LLM.
    def _build_expert_paths(self) -> None:
        self.logger.info("Starting expert paths analysis...")
        manager = ExpertPathManager(
            chunks_path=config.CHUNKS_METADATA_PATH,
            edges_path=config.GRAPH_EDGES_PATH,
            output_path=config.EXPERT_PATHS_PATH,
        )
        # Use new sequential API for better control and logging
        manager.prepare_data()
        manager.generate_catalog()
        manager.call_llm_for_paths()
        manager.process_paths()
        manager.save_paths()
        self.logger.info("Expert paths analysis completed")

    # Vectorize code chunks and build FAISS index for semantic retrieval.
    def _build_and_save_chunks_embeddings(self) -> None:
        self.logger.info("Building chunk embeddings...")
        nodes = utils.load_metadata(config.CHUNKS_METADATA_PATH)
        if not nodes:
            self.logger.error("No nodes extracted; cannot build FAISS index.")
            raise SystemExit("No nodes extracted; cannot build FAISS index.")

        self.logger.info(f"Processing {len(nodes)} nodes for embeddings")
        texts = [utils.chunk_to_text(node) for node in nodes]
        embeddings = self.model.encode(texts, batch_size=max(1, config.EMBEDDING_BATCH_SIZE), convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))

        node_hashes = {
            str(node["id"]): hashlib.sha256(text.encode("utf-8")).hexdigest()
            for node, text in zip(nodes, texts)
            if node.get("id") is not None
        }
        utils.save_json(
            config.EMBEDDING_STATE_PATH,
            {
                "model_name": config.EMBEDDING_MODEL_NAME,
                "node_count": len(nodes),
                "node_hashes": node_hashes,
            },
        )
        np.savez_compressed(
            str(config.EMBEDDINGS_CACHE_PATH),
            ids=np.array([int(node["id"]) for node in nodes], dtype=np.int64),
            embeddings=embeddings,
        )
        self.logger.info(f"Node embeddings built: {len(nodes)} nodes, dimension {dim}")

    # Vectorize community summaries and build FAISS index for high-level conceptual retrieval.
    def _build_and_save_community_embeddings(self) -> None:
        self.logger.info("Building community embeddings...")
        data = utils.load_json_object(config.COMMUNITIES_PATH, "communities.json")
        community_nodes = data.get("Community_Nodes", [])
        if not isinstance(community_nodes, list) or not community_nodes:
            self.logger.error("No Community_Nodes found; cannot build community FAISS index.")
            raise SystemExit("No Community_Nodes found; cannot build community FAISS index.")

        community_ids: List[str] = []
        texts: List[str] = []
        for item in community_nodes:
            if not isinstance(item, dict):
                continue
            cid = item.get("community_id")
            summary = str(item.get("summary", "")).strip()
            if cid is None or not summary:
                continue
            community_ids.append(str(cid))
            texts.append(summary)

        if not texts:
            self.logger.error("Community summaries are empty; cannot build community FAISS index.")
            raise SystemExit("Community summaries are empty; cannot build community FAISS index.")

        self.logger.info(f"Processing {len(texts)} community summaries for embeddings")
        embeddings = self.model.encode(texts, batch_size=max(1, config.EMBEDDING_BATCH_SIZE), convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, str(config.COMMUNITY_FAISS_INDEX_PATH))

        np.savez_compressed(
            str(config.COMMUNITY_EMBEDDINGS_CACHE_PATH),
            community_ids=np.array([int(cid) if str(cid).isdigit() else -1 for cid in community_ids], dtype=np.int64),
            embeddings=embeddings,
        )
        self.logger.info(f"Community embeddings built: {len(texts)} communities, dimension {dim}")

    # Vectorize community summaries and build FAISS index for high-level conceptual retrieval.
    def _build_and_save_community_embeddings(self) -> None:
        self.logger.info("Building community embeddings...")
        data = utils.load_json_object(config.COMMUNITIES_PATH, "communities.json")
        community_nodes = data.get("Community_Nodes", [])
        if not isinstance(community_nodes, list) or not community_nodes:
            self.logger.error("No Community_Nodes found; cannot build community FAISS index.")
            raise SystemExit("No Community_Nodes found; cannot build community FAISS index.")

        community_ids: List[str] = []
        texts: List[str] = []
        for item in community_nodes:
            if not isinstance(item, dict):
                continue
            cid = item.get("community_id")
            summary = str(item.get("summary", "")).strip()
            if cid is None or not summary:
                continue
            community_ids.append(str(cid))
            texts.append(summary)

        if not texts:
            self.logger.error("Community summaries are empty; cannot build community FAISS index.")
            raise SystemExit("Community summaries are empty; cannot build community FAISS index.")

        self.logger.info(f"Processing {len(texts)} community summaries for embeddings")
        embeddings = self.model.encode(texts, batch_size=max(1, config.EMBEDDING_BATCH_SIZE), convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, str(config.COMMUNITY_FAISS_INDEX_PATH))

        np.savez_compressed(
            str(config.COMMUNITY_EMBEDDINGS_CACHE_PATH),
            community_ids=np.array([int(cid) if str(cid).isdigit() else -1 for cid in community_ids], dtype=np.int64),
            embeddings=embeddings,
        )
        self.logger.info(f"Community embeddings built: {len(texts)} communities, dimension {dim}")



