import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env once so every script importing config shares the same variables.
# `override=True` ensures .env values are honored even if the shell has stale exports.
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Data directory and files
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_METADATA_PATH = DATA_DIR / "chunks_metadata.json"
GRAPH_EDGES_PATH = DATA_DIR / "graph_edges.json"
SEARCH_RESULTS_PATH = DATA_DIR / "search_results_with_graph.json"
COMPILE_COMMANDS_PATH = DATA_DIR / "compile_commands.json"
PROMPT_PATH = DATA_DIR / "prompt.md"
EMBEDDINGS_CACHE_PATH = DATA_DIR / "embeddings_cache.npz"
EMBEDDING_STATE_PATH = DATA_DIR / "embedding_state.json"
COMMUNITIES_PATH = DATA_DIR / "communities.json"
COMMUNITY_FAISS_INDEX_PATH = DATA_DIR / "community_faiss.index"
COMMUNITY_EMBEDDINGS_CACHE_PATH = DATA_DIR / "community_embeddings_cache.npz"
COMMUNITY_USE_STATIC_PATH_PARTITION = os.getenv("COMMUNITY_USE_STATIC_PATH_PARTITION", "false").lower() in {
	"1",
	"true",
	"yes",
	"on",
}

# Source directory
SRC_DIR = PROJECT_ROOT / "src" / "backend"
PROMPT_BUILDER_PATH = SRC_DIR / "prompt_builder.py"

# LLM Configuration
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_TOKEN = os.getenv("LLM_TOKEN")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_RESPONSE_LANGUAGE = os.getenv("LLM_RESPONSE_LANGUAGE", "Chinese")

# Embedding generation configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Community detection and module routing tunables
COMMUNITY_TARGET_MIN = int(os.getenv("COMMUNITY_TARGET_MIN", "8"))
COMMUNITY_TARGET_MAX = int(os.getenv("COMMUNITY_TARGET_MAX", "15"))
COMMUNITY_MIN_SIZE = int(os.getenv("COMMUNITY_MIN_SIZE", "3"))
COMMUNITY_GLOBAL_TOP_PERCENT = float(os.getenv("COMMUNITY_GLOBAL_TOP_PERCENT", "0.05"))
ROUTING_STRONG_MODULE_SIMILARITY = float(os.getenv("ROUTING_STRONG_MODULE_SIMILARITY", "0.35"))

# Query planner strategy templates and mappings.
# `template` supports a `{depth}` placeholder for dynamic traversal depth.
STRATEGY_CONFIG = {
	"valid_query_types": {
		"FUNCTION_BEHAVIOR",
		"EXECUTION_FLOW",
		"DATA_STRUCTURE",
		"RELATIONSHIP",
		"FILE_OR_MODULE",
		"UNKNOWN",
	},
	"strategy_templates": {
		"embedding_only": "embedding_only",
		"embedding_callees": "embedding + callees depth={depth}",
		"embedding_bfs": "embedding + bfs depth={depth}",
		"struct_reference_expansion": "struct_reference_expansion",
		"multi_seed_connection_search": "multi_seed_connection_search",
		"module_nodes_expansion": "module_nodes_expansion",
	},
	"type_to_strategy": {
		"FUNCTION_BEHAVIOR": {
			"template": "embedding_callees",
			"depth": 2,
		},
		"EXECUTION_FLOW": {
			"template": "embedding_bfs",
			"depth": 2,
		},
		"DATA_STRUCTURE": {
			"template": "struct_reference_expansion",
		},
		"RELATIONSHIP": {
			"template": "multi_seed_connection_search",
		},
		"FILE_OR_MODULE": {
			"template": "module_nodes_expansion",
		},
		"UNKNOWN": {
			"template": "embedding_only",
		},
	},
	# Keywords can raise traversal depth for specific query types.
	"keyword_depth_overrides": {
		"memory": {
			"FUNCTION_BEHAVIOR": 3,
			"EXECUTION_FLOW": 4,
		},
		"vm": {
			"FUNCTION_BEHAVIOR": 3,
			"EXECUTION_FLOW": 4,
		},
		"trap": {
			"FUNCTION_BEHAVIOR": 3,
			"EXECUTION_FLOW": 4,
		},
	},
}
