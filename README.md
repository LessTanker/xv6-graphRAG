# xv6-graphRAG

**A retrieval-and-reasoning assistant for the xv6-riscv kernel codebase**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd xv6-graphRAG

# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your LLM API credentials

# Rebuild the knowledge base
make rebuild

# Ask a question
make query Q="how does fork work in xv6?"

# Or use the web interface
make backend-server
make frontend-server
# Open http://127.0.0.1:3000
```

`xv6-graphRAG` is a retrieval-and-reasoning assistant for the `xv6-riscv` kernel codebase.
It combines vector retrieval with graph traversal, community detection, and LLM planning to answer system-level questions such as execution flow, subsystem responsibilities, and function interactions.

In short, this project helps you ask natural language questions about xv6 and get answers grounded in real source code evidence.

## How To Build

### Prerequisites

- Python `3.10+`
- Node.js `18+` (for Vite + React frontend)
- `bear` (for `compile_commands.json` generation)
- `libclang` runtime/dev package
- xv6 build toolchain available on your machine

Example (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y bear libclang-dev
```

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend (Vite + React + Tailwind)
cd src/frontend
npm install
cd ../..
```

### Full Rebuild

This command cleans `data/` and rebuilds everything (including `compile_commands.json`):

```bash
make rebuild
```

## How To Use

### 1. Configure `.env`

Create or edit `.env` at repo root:

```env
LLM_API_URL="https://api.deepseek.com/chat/completions"
LLM_TOKEN="your-token"
LLM_MODEL="deepseek-chat"

# Embedding model options
EMBEDDING_MODEL_NAME="BAAI/bge-small-en"
EMBEDDING_BATCH_SIZE="64"

# Default answer language for LLM outputs
LLM_RESPONSE_LANGUAGE="Chinese"

# false = graph-based community detection (default)
# true  = static path-based partitioning
COMMUNITY_USE_STATIC_PATH_PARTITION="false"
```

### 2. Ask a Question

```bash
make query Q="how does the trap path reach usertrap?"
```

If you change `LLM_RESPONSE_LANGUAGE` in `.env`, you do **not** need `make rebuild`.
Just run `make query ...` again in a new command invocation.

### 3. Use The Web UI

Start backend API and Vite frontend in two terminals:

```bash
make backend-server
make frontend-server
```

Then open `http://127.0.0.1:3000` in your browser.

Frontend API URL defaults to `http://127.0.0.1:8001/api/query`.
To override it, set environment variable before starting frontend:

```bash
cd src/frontend
VITE_API_URL="http://127.0.0.1:8002/api/query" npm run dev -- --host 127.0.0.1 --port 3000
```


## Pipeline And Tech Stack

![Pipeline](pipeline.png)

### GraphRAG Core

- Chunk extraction from xv6 source (`clang` AST based)
- Graph edge construction (`CALLS`, `USES_STRUCT`, `USES_GLOBAL`, `MACRO_USE`, etc.)
- Dense vector retrieval with `sentence-transformers` + `faiss`
- Graph expansion for structure-aware recall

### Community Detection

- Build directed/undirected graph from extracted edges
- Mark top 5% in-degree hubs as `GLOBAL_SHARED`
- Run Leiden (or static partition mode) for module communities
- Merge tiny communities and form stable module clusters
- Support multi-membership context via `node_memberships`

### HyDE + Module Routing

- HyDE generates a hypothetical answer to improve query embedding quality
- Query is matched against community summaries (`community_faiss.index`)
- `Global_Nodes.summary` is always mounted as mandatory context
- Planner outputs `restricted_community_id` when a strong module hit is detected
- If module hit is weak, retrieval falls back to full global graph

### Expert Path Generation

- Uses LLM to identify core xv6 call chains from function catalog
- Creates coherent learning paths focused on kernel behaviors
- Ensures path connectivity and removes utility functions (panic, printf, etc.)
- Covers key subsystems: scheduler, process lifecycle, syscall/trap path, memory management, filesystem I/O
- Outputs structured paths with themes, node IDs, and descriptions

## How It Is Implemented

### Main Components

- `src/backend/core/KnowledgeIndexer.py`
  - Offline indexing pipeline
  - Builds chunks, graph edges, base FAISS
  - Ensures community artifacts and community FAISS are ready

- `src/backend/core/CommunityManager.py`
  - Community detection and post-processing
  - Hierarchical summarization
  - Writes `data/communities.json`

- `src/backend/core/QueryProcessor.py`
  - HyDE generation
  - Query embedding
  - Planner JSON generation
  - Module routing and global-context injection

- `src/backend/core/GraphRetriever.py`
  - Base vector recall + graph traversal expansion
  - Community-restricted traversal when applicable
  - Full-graph fallback when no strong module route exists

### Expert Path Analysis

- `src/backend/core/ExpertPathManager.py`
  - Identifies core kernel call chains using LLM analysis
  - Generates expert learning paths for understanding xv6 subsystems
  - Validates path connectivity and removes utility function tails
  - Detects themes (scheduler, process, syscall_trap, memory, filesystem)
  - Provides modular API: `prepare_data()`, `generate_catalog()`, `call_llm_for_paths()`, `process_paths()`, `save_paths()`

- `src/backend/PipelineService.py`
  - Shared orchestration service used by both HTTP and CLI entrypoints

- `src/backend/WebApp.py`
  - HTTP API entrypoint for browser requests

### Key Data Artifacts (`data/`)

- `compile_commands.json`: compilation database generated from xv6 build
- `chunks_metadata.json`: extracted code chunks and metadata
- `graph_edges.json`: code relation graph edges
- `faiss.index`: chunk embedding index
- `communities.json`: global nodes + community nodes + memberships + summaries
- `community_faiss.index`: index built from community summaries
- `prompt.md`: final context passed to answer generation
- `expert_path.json`: expert learning paths for xv6 kernel understanding

## Make Commands

```bash
make help
make clean
make rebuild
make query Q="your question"
make backend-server
make frontend-server
```

## Recent Improvements

### ExpertPathManager Refactoring (March 2026)

The `ExpertPathManager` class has been refactored to follow modern software engineering practices:

1. **Modular Design**: Split monolithic `_extract_with_llm()` method into focused public methods:
   - `prepare_data()` - Loads chunks and builds graph structures
   - `generate_catalog()` - Creates function catalog for LLM
   - `call_llm_for_paths()` - Calls LLM API for initial paths
   - `process_paths()` - Validates and processes LLM-generated paths
   - `save_paths()` - Saves final paths to JSON

2. **Comprehensive Logging**: Added logging infrastructure similar to CommunityManager:
   - Dedicated log file: `log/backend/ExpertPathManager.log`
   - Detailed progress tracking at each pipeline stage
   - Error handling with appropriate log levels

3. **Configuration Management**: Moved hardcoded constants to class-level:
   - `MIN_CHAIN_LEN = 3`
   - `CATALOG_SIZE = 420`
   - `UTILITY_NAMES` set for filtering utility functions

4. **Improved Documentation**: Added CommunityManager-style docstrings to all methods with clear descriptions of parameters, return types, and purpose.

5. **Backward Compatibility**: Maintained `build_and_save()` wrapper for convenience while exposing granular API for better control.

This refactoring improves maintainability, debuggability, and consistency with the existing codebase architecture.

## License

[MIT License](LICENSE)
