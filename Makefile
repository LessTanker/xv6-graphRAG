# Makefile for OS-expert unified pipeline

PYTHON ?= .venv/bin/python
DATA_DIR = data
XV6_DIR = xv6-riscv

COMPILE_COMMANDS = $(DATA_DIR)/compile_commands.json
CHUNKS_METADATA = $(DATA_DIR)/chunks_metadata.json
GRAPH_EDGES = $(DATA_DIR)/graph_edges.json
FAISS_INDEX = $(DATA_DIR)/faiss.index
COMMUNITIES = $(DATA_DIR)/communities.json
COMMUNITY_FAISS = $(DATA_DIR)/community_faiss.index
SEARCH_RESULTS = $(DATA_DIR)/search_results_with_graph.json
PROMPT_FILE = $(DATA_DIR)/prompt.md
EMBED_STATE = $(DATA_DIR)/embedding_state.json
EMBED_CACHE = $(DATA_DIR)/embeddings_cache.npz
COMMUNITY_EMBED_CACHE = $(DATA_DIR)/community_embeddings_cache.npz

.PHONY: all compile-db index rebuild query clean clean-all clean-runtime help

all: index

compile-db: $(COMPILE_COMMANDS)

$(COMPILE_COMMANDS):
	@echo "Building $(XV6_DIR) to generate compile_commands.json..."
	@mkdir -p $(DATA_DIR)
	cd $(XV6_DIR) && make clean && bear -- make
	mv $(XV6_DIR)/compile_commands.json $(COMPILE_COMMANDS)

index: compile-db
	@echo "Running unified indexing pipeline..."
	$(PYTHON) -m src.main "index warmup" --rebuild-index

rebuild: clean index

query:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make query Q='how does scheduler work?'"; \
		exit 1; \
	fi
	$(PYTHON) -m src.main "$(Q)"

# Default clean strategy: remove generated runtime/index artifacts but keep compile DB.
clean: clean-runtime

clean-runtime:
	@echo "Cleaning runtime/index artifacts (keeping $(COMPILE_COMMANDS))..."
	rm -f $(CHUNKS_METADATA) $(GRAPH_EDGES) $(FAISS_INDEX)
	rm -f $(COMMUNITIES) $(COMMUNITY_FAISS)
	rm -f $(SEARCH_RESULTS) $(PROMPT_FILE)
	rm -f $(EMBED_STATE) $(EMBED_CACHE) $(COMMUNITY_EMBED_CACHE)

# Full clean strategy: also remove compile database.
clean-all: clean-runtime
	@echo "Also removing compile database..."
	rm -f $(COMPILE_COMMANDS)

help:
	@echo "Available targets:"
	@echo "  compile-db : Build xv6 and generate $(COMPILE_COMMANDS)"
	@echo "  index      : Rebuild chunks/graph/faiss using unified pipeline"
	@echo "  rebuild    : Clean then run full rebuild"
	@echo "  query      : Run query, example: make query Q='how does trap handling work?'"
	@echo "  clean      : Remove generated runtime/index artifacts (keep compile_commands.json)"
	@echo "  clean-all  : clean + remove compile_commands.json"
	@echo "  all        : Alias of index"
