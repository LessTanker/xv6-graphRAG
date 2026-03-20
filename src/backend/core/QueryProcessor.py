# Standard library imports
import json
import re
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Third-party library imports
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Local module imports
from backend import config, utils
from backend.core.LLMClient import LLMClient
from backend.logger import get_file_logger


# Configure logging for QueryProcessor using unified logger
logger = get_file_logger("backend.core.QueryProcessor")


@dataclass
class QueryBundle:
    """Container for processed query results including HyDE text, embedding, and plan."""
    raw_query: str
    hyde_text: str
    query_embedding: np.ndarray
    plan: Dict[str, Any]


class QueryProcessor:
    """
    Main query processing pipeline that handles HyDE generation, embedding, and planning.
    """

    _DYNAMIC_STRATEGY_PATTERN = re.compile(r"^embedding \+ (callees|bfs) depth=(\d+)$")
    _STRONG_MODULE_SIMILARITY = config.ROUTING_STRONG_MODULE_SIMILARITY

    def __init__(self, model: SentenceTransformer, response_language: str = config.LLM_RESPONSE_LANGUAGE):
        """Initialize QueryProcessor with embedding model and configuration."""
        self.model = model
        self.response_language = response_language
        self.llm_client = LLMClient(config.LLM_API_URL, config.LLM_TOKEN, config.LLM_MODEL)
        self.strategy_config = config.STRATEGY_CONFIG or {}
        self.valid_query_types = set(self.strategy_config.get("valid_query_types", {"UNKNOWN"}))
        self.strategy_templates = dict(self.strategy_config.get("strategy_templates", {}))
        self.type_to_strategy = dict(self.strategy_config.get("type_to_strategy", {}))
        self.keyword_depth_overrides = dict(self.strategy_config.get("keyword_depth_overrides", {}))

        self.valid_static_strategies = {
            template
            for template in self.strategy_templates.values()
            if isinstance(template, str) and "{depth}" not in template
        }

    # Main entry point for query processing.
    def process(self, query: str) -> QueryBundle:
        logger.info("Processing query: %s", query[:100] + "..." if len(query) > 100 else query)

        hyde_text = self._generate_hyde_hypothetical_document(query)
        query_embedding = self._build_hyde_query_embedding(query, hyde_text)
        plan = self._plan(query, query_embedding)

        logger.info("Query processing completed. Query type: %s, Strategy: %s",
                   plan.get("query_type"), plan.get("traversal_strategy"))

        return QueryBundle(raw_query=query, hyde_text=hyde_text, query_embedding=query_embedding, plan=plan)

    # Generate HyDE (Hypothetical Document Embedding) text for query enhancement.
    def _generate_hyde_hypothetical_document(self, query: str) -> str:
        if not self.llm_client.is_configured():
            raise RuntimeError("LLM client is not configured for HyDE generation. Please check LLM settings.")

        prompt = self._build_hyde_prompt(query)
        messages = self._build_hyde_messages(prompt)

        hyde_text = self.llm_client.chat(messages=messages, temperature=0.7, timeout=30, max_tokens=300)
        if not hyde_text:
            raise RuntimeError("HyDE generation returned empty content from LLM.")

        logger.debug("HyDE generation successful, length: %d chars", len(hyde_text))
        return hyde_text

    # Build query embedding by combining original query and HyDE text.
    def _build_hyde_query_embedding(self, query: str, hyde_doc: str) -> np.ndarray:
        query_instruct = "Represent this sentence for searching relevant passages: " + query
        hyde_instruct = "Represent this sentence for searching relevant passages: " + hyde_doc

        query_emb_orig = self.model.encode([query_instruct])[0].astype("float32")
        hyde_emb = self.model.encode([hyde_instruct])[0].astype("float32")
        return (0.3 * query_emb_orig + 0.7 * hyde_emb).reshape(1, -1)

    # Generate query plan with expert path matching and community routing.
    def _plan(self, query: str, query_embedding: np.ndarray) -> Dict[str, Any]:
        logger.info("Generating plan for query: %s", query[:100])

        # Step 1: Match expert paths if available
        expert_match = self._match_expert_path(query=query, query_embedding=query_embedding)
        if expert_match:
            logger.debug("Found expert path match: %s", expert_match.get("title"))

        # Step 2: Get relevant communities for module routing
        routed = self._get_relevant_communities(query_embedding=query_embedding)

        # Step 3: Generate plan using LLM (with built-in fallback)
        plan = self._generate_llm_plan(query, routed, expert_match)

        logger.info("Plan generated: type=%s, strategy=%s",
                   plan.get("query_type"), plan.get("traversal_strategy"))
        return plan

    # Build prompt for HyDE generation.
    def _build_hyde_prompt(self, query: str) -> str:
        return (
            "You are an expert in the xv6 operating system. "
            f"Always respond in {self.response_language}. "
            f"User Query: {query}\n"
            "Hypothetical Answer:"
        )
    
    # Build messages for HyDE generation.
    def _build_hyde_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a helpful assistant specializing in xv6 OS."},
            {"role": "user", "content": prompt},
        ]
    
    # Match query against expert learning paths.
    def _match_expert_path(self, query: str, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        expert_paths = self._load_expert_paths()
        if not expert_paths:
            return None

        texts, normalized_paths = self._prepare_expert_paths_for_matching(expert_paths)
        if not texts:
            return None

        path_embeddings = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        best_idx, best_score = self._find_best_expert_path_match(query_embedding, path_embeddings)

        if best_idx < 0 or best_score < config.EXPERT_PATH_MIN_SIMILARITY:
            return None

        matched = normalized_paths[best_idx]
        return self._build_expert_path_match_result(matched, best_score, query)

    # Load expert paths from JSON file.
    def _load_expert_paths(self) -> list:
        if not config.EXPERT_PATHS_PATH.exists():
            logger.debug("Expert paths file not found: %s", config.EXPERT_PATHS_PATH)
            return []

        try:
            data = utils.load_json_object(config.EXPERT_PATHS_PATH, "expert_path.json")
        except Exception as exc:
            logger.warning("Failed to load expert_path.json: %s", exc)
            return []

        paths = data.get("expert_paths", [])
        return paths if isinstance(paths, list) else []

    # Prepare expert paths for embedding matching.
    def _prepare_expert_paths_for_matching(self, expert_paths: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts = []
        normalized_paths = []

        for item in expert_paths:
            if not isinstance(item, dict):
                continue

            node_ids = item.get("node_ids", [])
            if not isinstance(node_ids, list) or len(node_ids) < 2:
                continue

            title = str(item.get("title", "")).strip()
            desc = str(item.get("description", "")).strip()
            node_names = item.get("node_names", [])
            if not isinstance(node_names, list):
                node_names = []

            text = (
                f"title: {title}\n"
                f"description: {desc}\n"
                f"chain: {' -> '.join(str(x) for x in node_names)}"
            )
            texts.append(text)
            normalized_paths.append(item)

        return texts, normalized_paths

    # Find best matching expert path using cosine similarity.
    def _find_best_expert_path_match(self, query_embedding: np.ndarray,
                                    path_embeddings: np.ndarray) -> Tuple[int, float]:
        query_vec = query_embedding.reshape(-1).astype("float32")
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0.0:
            return -1, -1.0

        best_idx = -1
        best_score = -1.0
        for idx, emb in enumerate(path_embeddings):
            denom = float(np.linalg.norm(emb)) * query_norm
            if denom == 0.0:
                continue
            score = float(np.dot(query_vec, emb) / denom)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx, best_score

    # Build expert path match result dictionary.
    def _build_expert_path_match_result(self, matched_path: Dict[str, Any],
                                       similarity: float, query: str) -> Dict[str, Any]:
        return {
            "id": str(matched_path.get("id", "")).strip(),
            "title": str(matched_path.get("title", "")).strip(),
            "description": str(matched_path.get("description", "")).strip(),
            "node_ids": [nid for nid in matched_path.get("node_ids", []) if isinstance(nid, int)],
            "node_names": [str(name) for name in matched_path.get("node_names", [])],
            "similarity": similarity,
            "query": query,
        }

    # Retrieve relevant communities for module routing.
    def _get_relevant_communities(self, query_embedding: np.ndarray, top_k: int = 3):
        if not config.COMMUNITY_FAISS_INDEX_PATH.exists() or not config.COMMUNITIES_PATH.exists():
            logger.debug("Community files not found, skipping community routing")
            return {"global_summary": "", "communities": []}

        try:
            index = faiss.read_index(str(config.COMMUNITY_FAISS_INDEX_PATH))
            communities_data = utils.load_json_object(config.COMMUNITIES_PATH, "communities.json")

            global_summary = self._extract_global_summary(communities_data)
            community_info = self._extract_community_info(communities_data)

            if not community_info["summaries"]:
                return {"global_summary": global_summary, "communities": []}

            ordered_ids = self._get_ordered_community_ids(community_info)
            k = min(max(1, top_k), len(ordered_ids))
            if k <= 0:
                return {"global_summary": global_summary, "communities": []}

            distances, indices = index.search(query_embedding, k=k)
            communities = self._build_community_results(distances, indices, ordered_ids, community_info)

            logger.debug("Found %d relevant communities", len(communities))
            return {"global_summary": global_summary, "communities": communities}

        except Exception as exc:
            logger.warning("Community routing lookup failed: %s", exc)
            return {"global_summary": "", "communities": []}

    # Extract global summary from communities data.
    def _extract_global_summary(self, communities_data: Dict[str, Any]) -> str:
        global_info = communities_data.get("Global_Nodes", {})
        if isinstance(global_info, dict):
            return str(global_info.get("summary", "")).strip()
        return ""

    # Extract community information from loaded data.
    def _extract_community_info(self, communities_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "summaries": communities_data.get("summaries", {}),
            "index_order": communities_data.get("index_order", []),
            "community_names": communities_data.get("community_names", {}),
        }

    # Get ordered list of community IDs.
    def _get_ordered_community_ids(self, community_info: Dict[str, Any]) -> List[str]:
        index_order = community_info["index_order"]
        if isinstance(index_order, list) and index_order:
            return [str(cid) for cid in index_order]
        else:
            # Fallback: sort numeric IDs
            summaries = community_info["summaries"]
            return sorted(
                [k for k in summaries.keys() if str(k).isdigit()],
                key=lambda x: int(x),
            )

    # Build community result dictionaries from search results.
    def _build_community_results(self, distances: np.ndarray, indices: np.ndarray,
                                ordered_ids: List[str], community_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        summaries = community_info["summaries"]
        community_names = community_info["community_names"]

        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(ordered_ids):
                continue

            cid = ordered_ids[idx]
            summary = str(summaries.get(cid, ""))
            distance = float(distances[0][rank])
            similarity = float(1 / (1 + distance))

            results.append({
                "community_id": int(cid) if str(cid).isdigit() else cid,
                "community_name": community_names.get(cid, f"Community {cid}") if isinstance(community_names, dict) else f"Community {cid}",
                "summary": summary,
                "similarity": similarity,
                "distance": distance,
                "is_global": False,
            })

        return results
    
    # Generate plan using LLM with community context.
    def _generate_llm_plan(self, query: str, routed: Dict[str, Any],
                          expert_match: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.llm_client.is_configured():
            raise RuntimeError("LLM client is not configured for query planning. Please check LLM settings.")

        community_context = self._build_community_context(routed)
        planner_prompt = self._build_planner_prompt(community_context=community_context)
        messages = self._build_planner_messages(planner_prompt, query)

        content = self.llm_client.chat(messages=messages, temperature=0.0, timeout=30)
        parsed = self._parse_json_object_from_text(content)

        if not parsed:
            raise RuntimeError("LLM returned invalid or empty JSON response for query planning.")

        plan = self._sanitize_plan(parsed, query)
        plan = self._apply_module_routing_bias(plan, routed)

        if expert_match is not None:
            plan["expert_path_match"] = expert_match

        return plan
    
    # Build community context string for planner prompt.
    def _build_community_context(self, routed: Dict[str, Any]) -> str:
        relevant_communities = routed.get("communities", []) if isinstance(routed, dict) else []
        global_summary = str(routed.get("global_summary", "")) if isinstance(routed, dict) else ""

        community_lines = []
        if global_summary:
            community_lines.append(
                "[MANDATORY_GLOBAL_CONTEXT] "
                "community_name=Global Kernel Utilities "
                f"summary={global_summary}"
            )

        for item in relevant_communities:
            community_name = item.get("community_name") or f"Community {item['community_id']}"
            community_lines.append(
                f"community_id={item['community_id']} "
                f"name={community_name} "
                f"similarity={item['similarity']:.4f} summary={item['summary']}"
            )

        return "\n".join(community_lines)
    
    # Build prompt for LLM query planner.
    def _build_planner_prompt(self, community_context: str = "") -> str:
        query_types = ", ".join(sorted(self.valid_query_types))
        strategy_examples = self._build_strategy_examples()
        mapping_lines = self._build_mapping_lines()

        return (
            "You are a query planner for xv6 code retrieval. Classify the user query and pick one traversal strategy. "
            f"Allowed query_type values: {query_types}. "
            "Depth in traversal_strategy is dynamic and should be a positive integer when applicable. "
            f"Allowed traversal_strategy values (examples): {', '.join(strategy_examples)}. "
            f"Default mapping must follow: {'; '.join(mapping_lines)}. "
            "Use the provided module community context to select one restricted community whenever possible. "
            "Return JSON only with keys: query_type, traversal_strategy, seed_selection_hint, reasoning, target_entity, restricted_community_id. "
            "restricted_community_id must be an integer community id from context, or null when uncertain. "
            f"\n\nModule community context:\n{community_context if community_context else 'N/A'}"
        )
    
    # Build messages for LLM planner.
    def _build_planner_messages(self, planner_prompt: str, query: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": f"{planner_prompt}\n\nUser query: {query}"},
        ]

    # Parse JSON object from LLM response text.
    def _parse_json_object_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        candidate = text.strip()
        candidate = self._extract_json_from_code_blocks(candidate)

        try:
            return json.loads(candidate)
        except Exception:
            return self._extract_json_by_brackets(candidate)
        
    # Extract JSON from code blocks (```json or ```).
    def _extract_json_from_code_blocks(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        return text

    # Extract JSON by finding outermost braces.
    def _extract_json_by_brackets(self, text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    # Sanitize and validate LLM-generated plan.
    def _sanitize_plan(self, plan: Optional[Dict[str, Any]], query_text: str) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError(f"LLM returned invalid plan format. Expected dict, got {type(plan).__name__}.")

        query_type = self._sanitize_query_type(plan)
        strategy = self._sanitize_traversal_strategy(plan, query_type, query_text)
        restricted_community_id = self._sanitize_restricted_community_id(plan)

        return {
            "query_type": query_type,
            "traversal_strategy": strategy,
            "seed_selection_hint": str(plan.get("seed_selection_hint", "embedding top results")).strip(),
            "reasoning": str(plan.get("reasoning", "")).strip(),
            "target_entity": str(plan.get("target_entity", "")).strip(),
            "restricted_community_id": restricted_community_id,
            "raw_query": query_text,
        }
    
    # Sanitize query type from plan.
    def _sanitize_query_type(self, plan: Dict[str, Any]) -> str:
        query_type = str(plan.get("query_type", "UNKNOWN")).strip().upper()
        if query_type not in self.valid_query_types:
            return "UNKNOWN"
        return query_type

    # Sanitize traversal strategy from plan.
    def _sanitize_traversal_strategy(self, plan: Dict[str, Any], query_type: str, query_text: str) -> str:
        strategy = str(plan.get("traversal_strategy", "")).strip()
        if not strategy:
            raise ValueError(f"LLM plan missing traversal_strategy for query_type: {query_type}")

        if not self._is_valid_strategy(strategy):
            raise ValueError(f"LLM returned invalid traversal_strategy: '{strategy}' for query_type: {query_type}")

        return self._apply_domain_depth_to_strategy(strategy, query_type, query_text)

    # Sanitize restricted community ID from plan.
    def _sanitize_restricted_community_id(self, plan: Dict[str, Any]) -> Optional[int]:
        restricted_raw = plan.get("restricted_community_id")
        if isinstance(restricted_raw, int):
            return restricted_raw
        elif isinstance(restricted_raw, str) and restricted_raw.strip().isdigit():
            return int(restricted_raw.strip())
        return None
    
    # Apply module routing bias to plan based on community similarity.
    def _apply_module_routing_bias(self, plan: Dict[str, Any], routed: Dict[str, Any]) -> Dict[str, Any]:
        communities = routed.get("communities", []) if isinstance(routed, dict) else []
        if not communities:
            return plan

        top_community = communities[0]
        similarity = float(top_community.get("similarity", 0.0))
        cid = top_community.get("community_id")

        if not self._should_apply_routing_bias(similarity, cid):
            return plan

        plan = self._apply_routing_to_plan(plan, cid, similarity)
        return plan
    
    # Check if routing bias should be applied based on similarity threshold.
    def _should_apply_routing_bias(self, similarity: float, community_id: Any) -> bool:
        if not isinstance(community_id, int):
            return False
        return similarity >= self._STRONG_MODULE_SIMILARITY
    
    # Apply routing bias to query plan.
    def _apply_routing_to_plan(self, plan: Dict[str, Any], community_id: int, similarity: float) -> Dict[str, Any]:
        # Set restricted community ID
        if plan.get("restricted_community_id") is None:
            plan["restricted_community_id"] = community_id

        # Update query type if uncertain
        current_type = str(plan.get("query_type", "UNKNOWN")).upper()
        if current_type in {"UNKNOWN", "FILE_OR_MODULE"}:
            plan["query_type"] = "FILE_OR_MODULE"

        # Enhance traversal strategy
        strategy = str(plan.get("traversal_strategy", "embedding_only"))
        plan["traversal_strategy"] = self._enhance_traversal_strategy(strategy)

        # Update seed selection hint
        hint = str(plan.get("seed_selection_hint", "embedding top results")).strip()
        plan["seed_selection_hint"] = f"{hint}; strong module routing to community_id={community_id}"

        logger.debug("Applied module routing bias to community_id=%d (similarity=%.4f)", community_id, similarity)
        return plan
    
    # Enhance traversal strategy for module routing.
    def _enhance_traversal_strategy(self, strategy: str) -> str:
        match = self._DYNAMIC_STRATEGY_PATTERN.match(strategy)
        if match:
            mode, depth_text = match.groups()
            depth = max(1, int(depth_text)) + 1
            return f"embedding + {mode} depth={depth}"
        elif strategy == "embedding_only":
            return "embedding + bfs depth=3"
        elif strategy == "module_nodes_expansion":
            return "embedding + bfs depth=3"
        return strategy
    
    # Get strategy for given query type with keyword overrides applied.
    # Validate strategy string format.
    def _is_valid_strategy(self, strategy: str) -> bool:
        if strategy in self.valid_static_strategies:
            return True
        match = self._DYNAMIC_STRATEGY_PATTERN.match(strategy or "")
        return bool(match and int(match.group(2)) >= 1)

    # Generate fallback plan when LLM planning fails.
    # Render strategy template with optional depth parameter.
    def _render_strategy(self, template_key: str, depth: Optional[int] = None) -> str:
        template = self.strategy_templates.get(template_key, "embedding_only")
        if "{depth}" in template:
            return template.format(depth=max(1, int(depth or 1)))
        return template

    # Apply keyword-based depth overrides to strategy.
    def _apply_keyword_depth_override(self, query_text: str, query_type: str, base_depth: Optional[int]) -> Optional[int]:
        if base_depth is None:
            return base_depth

        best_depth = int(base_depth)
        lowered = (query_text or "").lower()
        for keyword, type_overrides in self.keyword_depth_overrides.items():
            if keyword.lower() not in lowered:
                continue
            candidate = type_overrides.get(query_type)
            if candidate is None:
                continue
            best_depth = max(best_depth, int(candidate))
        return best_depth

    # Apply domain-specific depth adjustments to strategy.
    def _apply_domain_depth_to_strategy(self, strategy: str, query_type: str, query_text: str) -> str:
        match = self._DYNAMIC_STRATEGY_PATTERN.match(strategy or "")
        if not match:
            return strategy

        mode, depth_text = match.groups()
        current_depth = int(depth_text)
        target_depth = self._apply_keyword_depth_override(query_text, query_type, current_depth)
        if target_depth is None or target_depth <= current_depth:
            return strategy

        template_key = "embedding_callees" if mode == "callees" else "embedding_bfs"
        return self._render_strategy(template_key, target_depth)

    # Build strategy examples for planner prompt.
    def _build_strategy_examples(self) -> List[str]:
        strategy_examples = []
        for template in self.strategy_templates.values():
            if "{depth}" in template:
                strategy_examples.append(template.format(depth=3))
            else:
                strategy_examples.append(template)
        return strategy_examples

    # Build mapping lines for planner prompt.
    def _build_mapping_lines(self) -> List[str]:
        mapping_lines = []
        for query_type, mapping in self.type_to_strategy.items():
            rendered = self._render_strategy(mapping.get("template", "embedding_only"), mapping.get("depth"))
            mapping_lines.append(f"{query_type}->{rendered}")
        return mapping_lines