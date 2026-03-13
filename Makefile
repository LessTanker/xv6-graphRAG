# Makefile for OS-expert unified pipeline

PYTHON ?= .venv/bin/python
DATA_DIR = data
XV6_DIR = xv6-riscv

COMPILE_COMMANDS = $(DATA_DIR)/compile_commands.json
CHUNKS_METADATA = $(DATA_DIR)/chunks_metadata.json
GRAPH_EDGES = $(DATA_DIR)/graph_edges.json
FAISS_INDEX = $(DATA_DIR)/faiss.index
SEARCH_RESULTS = $(DATA_DIR)/search_results_with_graph.json
PROMPT_FILE = $(DATA_DIR)/prompt.md

.PHONY: all compile-db index rebuild query clean help

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

clean:
	rm -f $(CHUNKS_METADATA) $(GRAPH_EDGES) $(FAISS_INDEX) $(SEARCH_RESULTS) $(PROMPT_FILE)
	rm -f $(DATA_DIR)/embedding_state.json $(DATA_DIR)/embeddings_cache.npz

help:
	@echo "Available targets:"
	@echo "  compile-db : Build xv6 and generate $(COMPILE_COMMANDS)"
	@echo "  index      : Rebuild chunks/graph/faiss using unified pipeline"
	@echo "  rebuild    : Clean then run full rebuild"
	@echo "  query      : Run query, example: make query Q='how does trap handling work?'"
	@echo "  clean      : Remove generated data artifacts"
	@echo "  all        : Alias of index"
