import argparse
import contextlib
import io
import logging
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

try:
    from src import config
    from src.GraphRetriever import GraphRetriever
    from src.KnowledgeIndexer import KnowledgeIndexer
    from src.QueryProcessor import QueryProcessor
    from src.ResponseGenerator import ResponseGenerator
except ImportError:
    import config  # type: ignore
    from GraphRetriever import GraphRetriever  # type: ignore
    from KnowledgeIndexer import KnowledgeIndexer  # type: ignore
    from QueryProcessor import QueryProcessor  # type: ignore
    from ResponseGenerator import ResponseGenerator  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified GraphRAG pipeline for xv6.")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild chunks, graph, and FAISS index before querying",
    )
    return parser.parse_args()


def configure_runtime() -> None:
    load_dotenv(config.PROJECT_ROOT / ".env")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def load_embedding_model_silent(model_name: str) -> SentenceTransformer:
    # Some backends print progress bars directly to stdout/stderr during model load.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return SentenceTransformer(model_name)


def main() -> None:
    args = parse_args()
    configure_runtime()

    model = load_embedding_model_silent(config.EMBEDDING_MODEL_NAME)

    indexer = KnowledgeIndexer(source_root=config.PROJECT_ROOT, model=model)
    indexer.ensure_ready(force_rebuild=args.rebuild_index)

    query_processor = QueryProcessor(model=model)
    query_bundle = query_processor.process(args.query)

    retriever = GraphRetriever(model=model)
    retrieval = retriever.retrieve(query_bundle.query_embedding, query_bundle.plan)

    responder = ResponseGenerator()
    final_output = responder.generate(
        query=query_bundle.raw_query,
        plan=query_bundle.plan,
        seeds=retrieval.seeds,
        related_chunks=retrieval.related_chunks,
    )

    top_count = len(final_output.get("top3_nodes", []))
    related_count = len(final_output.get("directly_related_nodes", []))
    print(f"Saved combined retrieval results to {config.SEARCH_RESULTS_PATH}")
    print(f"Top-3 nodes: {top_count}, directly related nodes: {related_count}")
    print("\nLLM Response:\n")
    print(final_output.get("llm_response", ""))


if __name__ == "__main__":
    main()
