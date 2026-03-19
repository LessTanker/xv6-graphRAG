"""Backend package for GraphRAG-based code analysis system."""

# Import config directly
from . import config

# Import core modules
from .core import (
    CommunityManager,
    ExpertPathManager,
    GraphRetriever,
    KnowledgeIndexer,
    QueryProcessor,
    ResponseGenerator,
    LLMClient,
)

__all__ = [
    "CommunityManager",
    "ExpertPathManager",
    "GraphRetriever",
    "KnowledgeIndexer",
    "QueryProcessor",
    "ResponseGenerator",
    "LLMClient",
    "config",
    "utils",
]