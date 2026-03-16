import argparse
import os

try:
    from backend import config
    from backend.PipelineService import pipeline_service
except ImportError:
    import config  # type: ignore
    from PipelineService import pipeline_service  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified GraphRAG pipeline for xv6.")
    parser.add_argument("query", nargs="?", help="Natural language query")
    parser.add_argument(
        "--answer-language",
        default=None,
        help="Language for LLM outputs (defaults to LLM_RESPONSE_LANGUAGE from .env)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild chunks, graph, and FAISS index before querying",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only (re)build index artifacts and exit without running query/LLM",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answer_language = args.answer_language or os.getenv("LLM_RESPONSE_LANGUAGE", config.LLM_RESPONSE_LANGUAGE)
    print(f"Using answer language: {answer_language}")

    if not args.index_only and not args.query:
        raise SystemExit("query is required unless --index-only is set")

    pipeline_service.ensure_initialized(rebuild_index=args.rebuild_index)

    if args.index_only:
        print("Index build complete. Skipped query processing and LLM answering (--index-only).")
        return

    final_output = pipeline_service.answer_query(
        query=args.query or "",
        answer_language=answer_language,
    )

    top_count = len(final_output.get("top3_nodes", []))
    related_count = len(final_output.get("directly_related_nodes", []))
    print(f"Saved combined retrieval results to {config.SEARCH_RESULTS_PATH}")
    print(f"Top-3 nodes: {top_count}, directly related nodes: {related_count}")
    print("\nLLM Response:\n")
    print(final_output.get("llm_response", ""))


if __name__ == "__main__":
    main()
