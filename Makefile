PYTHON ?= .venv/bin/python
DATA_DIR := data
XV6_DIR := xv6-riscv

.PHONY: clean rebuild query help

clean:
	@echo "Cleaning $(DATA_DIR)/ ..."
	@mkdir -p $(DATA_DIR)
	@find $(DATA_DIR) -mindepth 1 -maxdepth 1 -exec rm -rf {} +

rebuild: clean
	@echo "Rebuilding compile database and all indexes..."
	@mkdir -p $(DATA_DIR)
	@cd $(XV6_DIR) && make clean && bear -- make
	@mv $(XV6_DIR)/compile_commands.json $(DATA_DIR)/compile_commands.json
	@$(PYTHON) -m src.main --rebuild-index --index-only

query:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make query Q='how does scheduler work?'"; \
		exit 1; \
	fi
	@$(PYTHON) -m src.main "$(Q)"

help:
	@echo "Available targets:"
	@echo "  make clean                 Remove all files under data/"
	@echo "  make rebuild               Rebuild compile_commands.json and regenerate all data artifacts"
	@echo "  make query Q=\"...\"        Run a question through the full online pipeline"
	@echo "  make help                  Show this help"
