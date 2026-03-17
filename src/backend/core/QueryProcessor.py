import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from backend import config, utils
except ImportError:
    import config  # type: ignore
    import utils  # type: ignore


DEFAULT_PLAN = {
    "query_type": "UNKNOWN",
    "traversal_strategy": "embedding_only",
    "seed_selection_hint": "embedding top results",
    "reasoning": "Fallback because planner result was unavailable.",
    "target_entity": "",
    "restricted_community_id": None,
}


@dataclass
class QueryBundle:
    raw_query: str
    hyde_text: str
    query_embedding: np.ndarray
    plan: Dict[str, Any]


class LLMClient:
    def __init__(self, api_url: Optional[str], api_key: Optional[str], model: Optional[str]):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_url and self.api_key and self.model)

    def chat(self, messages, temperature: float = 0.0, timeout: int = 30) -> str:
        if not self.is_configured():
            return ""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        return data.get("choices", [])[0].get("message", {}).get("content", "")


class QueryProcessor:
    _DYNAMIC_STRATEGY_PATTERN = re.compile(r"^embedding \+ (callees|bfs) depth=(\d+)$")
    _STRONG_MODULE_SIMILARITY = config.ROUTING_STRONG_MODULE_SIMILARITY

    def __init__(self, model: SentenceTransformer, response_language: str = config.LLM_RESPONSE_LANGUAGE):
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

    def process(self, query: str) -> QueryBundle:
        hyde_text = self.generate_hyde_hypothetical_document(query)
        query_embedding = self.build_hyde_query_embedding(query, hyde_text)
        plan = self.plan(query, query_embedding)
        return QueryBundle(raw_query=query, hyde_text=hyde_text, query_embedding=query_embedding, plan=plan)

    def generate_hyde_hypothetical_document(self, query: str) -> str:
        if not self.llm_client.is_configured():
            return query

        prompt = (
            "You are an expert in the xv6 operating system. "
            f"Always respond in {self.response_language}. "
            f"User Query: {query}\n"
            "Hypothetical Answer:"
        )

        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specializing in xv6 OS."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        }

        req = urllib.request.Request(
            config.LLM_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.LLM_TOKEN}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            content = data.get("choices", [])[0].get("message", {}).get("content", "")
            return content.strip() if content else query
        except Exception as exc:
            logging.error("HyDE generation failed: %s", exc)
            return query

    def build_hyde_query_embedding(self, query: str, hyde_doc: str) -> np.ndarray:
        query_instruct = "Represent this sentence for searching relevant passages: " + query
        hyde_instruct = "Represent this sentence for searching relevant passages: " + hyde_doc

        query_emb_orig = self.model.encode([query_instruct])[0].astype("float32")
        hyde_emb = self.model.encode([hyde_instruct])[0].astype("float32")
        return (0.3 * query_emb_orig + 0.7 * hyde_emb).reshape(1, -1)

    def _render_strategy(self, template_key: str, depth: Optional[int] = None) -> str:
        template = self.strategy_templates.get(template_key, "embedding_only")
        if "{depth}" in template:
            return template.format(depth=max(1, int(depth or 1)))
        return template

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

    def _strategy_from_query_type(self, query_type: str, query_text: str) -> str:
        mapping = self.type_to_strategy.get(query_type, self.type_to_strategy.get("UNKNOWN", {"template": "embedding_only"}))
        template_key = mapping.get("template", "embedding_only")
        depth = self._apply_keyword_depth_override(query_text, query_type, mapping.get("depth"))
        return self._render_strategy(template_key, depth)

    def _is_valid_strategy(self, strategy: str) -> bool:
        if strategy in self.valid_static_strategies:
            return True
        match = self._DYNAMIC_STRATEGY_PATTERN.match(strategy or "")
        return bool(match and int(match.group(2)) >= 1)

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

    def _get_relevant_communities(self, query_embedding: np.ndarray, top_k: int = 3):
        if not config.COMMUNITY_FAISS_INDEX_PATH.exists() or not config.COMMUNITIES_PATH.exists():
            return {"global_summary": "", "communities": []}

        try:
            index = faiss.read_index(str(config.COMMUNITY_FAISS_INDEX_PATH))
            communities_data = utils.load_json_object(config.COMMUNITIES_PATH, "communities.json")
            summaries = communities_data.get("summaries", {})
            index_order = communities_data.get("index_order", [])
            community_names = communities_data.get("community_names", {})
            global_info = communities_data.get("Global_Nodes", {})
            global_summary = ""
            if isinstance(global_info, dict):
                global_summary = str(global_info.get("summary", "")).strip()

            if not isinstance(summaries, dict) or not summaries:
                return {"global_summary": global_summary, "communities": []}

            if isinstance(index_order, list) and index_order:
                ordered_ids = [str(cid) for cid in index_order]
            else:
                ordered_ids = sorted(
                    [k for k in summaries.keys() if str(k).isdigit()],
                    key=lambda x: int(x),
                )

            k = min(max(1, top_k), len(ordered_ids))
            if k <= 0:
                return {"global_summary": global_summary, "communities": []}

            distances, indices = index.search(query_embedding, k=k)

            results = []
            for rank, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(ordered_ids):
                    continue
                cid = ordered_ids[idx]
                summary = str(summaries.get(cid, ""))
                distance = float(distances[0][rank])
                similarity = float(1 / (1 + distance))
                results.append(
                    {
                        "community_id": int(cid) if str(cid).isdigit() else cid,
                        "community_name": community_names.get(cid, f"Community {cid}") if isinstance(community_names, dict) else f"Community {cid}",
                        "summary": summary,
                        "similarity": similarity,
                        "distance": distance,
                        "is_global": False,
                    }
                )
            return {"global_summary": global_summary, "communities": results}
        except Exception as exc:
            logging.warning("Community routing lookup failed: %s", exc)
            return {"global_summary": "", "communities": []}

    def _apply_module_routing_bias(self, plan: Dict[str, Any], routed: Dict[str, Any]) -> Dict[str, Any]:
        communities = routed.get("communities", []) if isinstance(routed, dict) else []
        if not communities:
            return plan

        top = communities[0]
        similarity = float(top.get("similarity", 0.0))
        cid = top.get("community_id")
        if not isinstance(cid, int):
            return plan

        if similarity < self._STRONG_MODULE_SIMILARITY:
            return plan

        if plan.get("restricted_community_id") is None:
            plan["restricted_community_id"] = cid

        current_type = str(plan.get("query_type", "UNKNOWN")).upper()
        if current_type in {"UNKNOWN", "FILE_OR_MODULE"}:
            plan["query_type"] = "FILE_OR_MODULE"

        strategy = str(plan.get("traversal_strategy", "embedding_only"))
        match = self._DYNAMIC_STRATEGY_PATTERN.match(strategy)
        if match:
            mode, depth_text = match.groups()
            depth = max(1, int(depth_text)) + 1
            plan["traversal_strategy"] = f"embedding + {mode} depth={depth}"
        elif strategy == "embedding_only":
            plan["traversal_strategy"] = "embedding + bfs depth=3"
        elif strategy == "module_nodes_expansion":
            plan["traversal_strategy"] = "embedding + bfs depth=3"

        hint = str(plan.get("seed_selection_hint", "embedding top results")).strip()
        plan["seed_selection_hint"] = (
            f"{hint}; strong module routing to community_id={cid}"
        )
        return plan

    def _load_expert_paths(self) -> list:
        if not config.EXPERT_PATHS_PATH.exists():
            return []
        try:
            data = utils.load_json_object(config.EXPERT_PATHS_PATH, "expert_path.json")
        except Exception as exc:
            logging.warning("Failed to load expert_path.json: %s", exc)
            return []
        paths = data.get("expert_paths", [])
        return paths if isinstance(paths, list) else []

    def _match_expert_path(self, query: str, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        expert_paths = self._load_expert_paths()
        if not expert_paths:
            return None

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

        if not texts:
            return None

        path_embeddings = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        query_vec = query_embedding.reshape(-1).astype("float32")
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0.0:
            return None

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

        if best_idx < 0 or best_score < config.EXPERT_PATH_MIN_SIMILARITY:
            return None

        matched = normalized_paths[best_idx]
        return {
            "id": str(matched.get("id", "")).strip(),
            "title": str(matched.get("title", "")).strip(),
            "description": str(matched.get("description", "")).strip(),
            "node_ids": [nid for nid in matched.get("node_ids", []) if isinstance(nid, int)],
            "node_names": [str(name) for name in matched.get("node_names", [])],
            "similarity": best_score,
            "query": query,
        }

    def build_planner_prompt(self, community_context: str = "") -> str:
        query_types = ", ".join(sorted(self.valid_query_types))
        strategy_examples = []
        for template in self.strategy_templates.values():
            if "{depth}" in template:
                strategy_examples.append(template.format(depth=3))
            else:
                strategy_examples.append(template)

        mapping_lines = []
        for query_type, mapping in self.type_to_strategy.items():
            rendered = self._render_strategy(mapping.get("template", "embedding_only"), mapping.get("depth"))
            mapping_lines.append(f"{query_type}->{rendered}")

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

    def parse_json_object_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        candidate = text.strip()
        if "```json" in candidate:
            candidate = candidate.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in candidate:
            candidate = candidate.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            return json.loads(candidate)
        except Exception:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except Exception:
                    return None
        return None

    def sanitize_plan(self, plan: Optional[Dict[str, Any]], query_text: str) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            return self.fallback_plan(query_text)

        query_type = str(plan.get("query_type", "UNKNOWN")).strip().upper()
        if query_type not in self.valid_query_types:
            query_type = "UNKNOWN"

        fallback_strategy = self._strategy_from_query_type(query_type, query_text)
        strategy = str(plan.get("traversal_strategy", "")).strip()
        if not self._is_valid_strategy(strategy):
            strategy = fallback_strategy
        strategy = self._apply_domain_depth_to_strategy(strategy, query_type, query_text)

        restricted_raw = plan.get("restricted_community_id")
        if isinstance(restricted_raw, int):
            restricted_community_id = restricted_raw
        elif isinstance(restricted_raw, str) and restricted_raw.strip().isdigit():
            restricted_community_id = int(restricted_raw.strip())
        else:
            restricted_community_id = None

        return {
            "query_type": query_type,
            "traversal_strategy": strategy,
            "seed_selection_hint": str(plan.get("seed_selection_hint", "embedding top results")).strip(),
            "reasoning": str(plan.get("reasoning", "")).strip(),
            "target_entity": str(plan.get("target_entity", "")).strip(),
            "restricted_community_id": restricted_community_id,
            "raw_query": query_text,
        }

    def fallback_plan(self, query_text: str) -> Dict[str, Any]:
        query_type = "UNKNOWN"
        return {
            **DEFAULT_PLAN,
            "query_type": query_type,
            "traversal_strategy": self._strategy_from_query_type(query_type, query_text),
            "raw_query": query_text,
        }

    def plan(self, query: str, query_embedding: np.ndarray) -> Dict[str, Any]:
        expert_match = self._match_expert_path(query=query, query_embedding=query_embedding)

        routed = self._get_relevant_communities(query_embedding=query_embedding)
        relevant_communities = routed.get("communities", []) if isinstance(routed, dict) else []
        global_summary = str(routed.get("global_summary", "")) if isinstance(routed, dict) else ""

        if not self.llm_client.is_configured():
            fallback = self.fallback_plan(query)
            plan = self._apply_module_routing_bias(fallback, routed)
            if expert_match is not None:
                plan["expert_path_match"] = expert_match
            return plan

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
        planner_prompt = self.build_planner_prompt(community_context="\n".join(community_lines))
        messages = [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": f"{planner_prompt}\n\nUser query: {query}"},
        ]

        try:
            content = self.llm_client.chat(messages=messages, temperature=0.0, timeout=30)
            parsed = self.parse_json_object_from_text(content)
            plan = self.sanitize_plan(parsed, query)
            plan = self._apply_module_routing_bias(plan, routed)
            if expert_match is not None:
                plan["expert_path_match"] = expert_match
            return plan
        except Exception as exc:
            logging.error("Planner LLM call failed: %s", exc)
            fallback = self.fallback_plan(query)
            plan = self._apply_module_routing_bias(fallback, routed)
            if expert_match is not None:
                plan["expert_path_match"] = expert_match
            return plan
