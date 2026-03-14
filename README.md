# xv6-graphRAG

## What This Project Is

`xv6-graphRAG` is a retrieval-and-reasoning assistant for the `xv6-riscv` kernel codebase.
It combines vector retrieval with graph traversal, community detection, and LLM planning to answer system-level questions such as execution flow, subsystem responsibilities, and function interactions.

In short, this project helps you ask natural language questions about xv6 and get answers grounded in real source code evidence.

## How To Build

### Prerequisites

- Python `3.10+`
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

### On-demand Summary Generation

- CommunityManager generates:
  - global shared utility summary
  - per-community module summaries (English)
- GraphRetriever can enrich missing chunk summaries on demand and refresh FAISS index

## How It Is Implemented

### Main Components

- `src/KnowledgeIndexer.py`
  - Offline indexing pipeline
  - Builds chunks, graph edges, base FAISS
  - Ensures community artifacts and community FAISS are ready

- `src/CommunityManager.py`
  - Community detection and post-processing
  - Hierarchical summarization
  - Writes `data/communities.json`

- `src/QueryProcessor.py`
  - HyDE generation
  - Query embedding
  - Planner JSON generation
  - Module routing and global-context injection

- `src/GraphRetriever.py`
  - Base vector recall + graph traversal expansion
  - Community-restricted traversal when applicable
  - Full-graph fallback when no strong module route exists

- `src/ResponseGenerator.py`
  - Prompt assembly
  - Final LLM answer generation
  - Writes final retrieval/answer output files

### Key Data Artifacts (`data/`)

- `compile_commands.json`: compilation database generated from xv6 build
- `chunks_metadata.json`: extracted code chunks and metadata
- `graph_edges.json`: code relation graph edges
- `faiss.index`: chunk embedding index
- `communities.json`: global nodes + community nodes + memberships + summaries
- `community_faiss.index`: index built from community summaries
- `prompt.md`: final context passed to answer generation
- `search_results_with_graph.json`: final retrieval + plan + answer payload

## Make Commands

```bash
make help
make clean
make rebuild
make query Q="your question"
```

## License

[MIT License](LICENSE)
