# Standard library imports
import contextlib
import io
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

# Third-party library imports
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

# Local module imports
from backend import config
from backend.core import GraphRetriever, KnowledgeIndexer, QueryProcessor, ResponseGenerator


class PipelineService:
    """Thread-safe wrapper for loading model/index once and answering queries."""

    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False
        self._model: Optional[SentenceTransformer] = None

    def _configure_runtime(self) -> None:
        load_dotenv(config.PROJECT_ROOT / ".env", override=True)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TQDM_DISABLE"] = "1"
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    def _load_embedding_model_silent(self, model_name: str) -> SentenceTransformer:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return SentenceTransformer(model_name)

    def ensure_initialized(self, rebuild_index: bool = False) -> None:
        with self._lock:
            if self._initialized and not rebuild_index:
                return

            self._configure_runtime()
            self._model = self._load_embedding_model_silent(config.EMBEDDING_MODEL_NAME)

            indexer = KnowledgeIndexer(source_root=config.PROJECT_ROOT, model=self._model)
            indexer.ensure_ready(force_rebuild=rebuild_index)
            self._initialized = True

    def answer_query(
        self,
        query: str,
        answer_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")

        self.ensure_initialized()
        model = self._model
        if model is None:
            raise RuntimeError("pipeline model failed to initialize")

        language = answer_language or os.getenv("LLM_RESPONSE_LANGUAGE", config.LLM_RESPONSE_LANGUAGE)

        query_processor = QueryProcessor(model=model, response_language=language)
        query_bundle = query_processor.process(query)

        retriever = GraphRetriever(model=model)
        retrieval = retriever.retrieve(query_bundle.query_embedding, query_bundle.plan)

        responder = ResponseGenerator()
        return responder.generate(
            query=query_bundle.raw_query,
            plan=query_bundle.plan,
            seeds=retrieval.seeds,
            related_chunks=retrieval.related_chunks,
            answer_language=language,
        )


pipeline_service = PipelineService()
