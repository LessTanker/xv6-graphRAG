# Standard library imports
import json
import logging
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Local module imports
from backend import config


class LLMClient:
    """Unified client for making LLM API calls with comprehensive functionality."""

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLMClient with configuration.

        Args:
            api_url: LLM API endpoint URL (defaults to config.LLM_API_URL)
            api_key: LLM API key/token (defaults to config.LLM_TOKEN)
            model: LLM model name (defaults to config.LLM_MODEL)
        """
        self.api_url = api_url or config.LLM_API_URL
        self.api_key = api_key or config.LLM_TOKEN
        self.model = model or config.LLM_MODEL

        # Configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler
        log_dir = config.PROJECT_ROOT / "log" / "backend"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "LLMClient.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)

        self.logger.info("LLMClient initialized with model: %s", self.model)

    def is_configured(self) -> bool:
        """Check if LLM client is properly configured.

        Returns:
            True if all required configuration is present
        """
        return bool(self.api_url and self.api_key and self.model)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0,
             timeout: int = 30, max_tokens: Optional[int] = None) -> str:
        """Send chat completion request to LLM API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text content from LLM

        Raises:
            RuntimeError: If client is not configured
            ValueError: If messages are invalid
            urllib.error.HTTPError: If API request fails
        """
        if not self.is_configured():
            raise RuntimeError(
                "LLM client is not properly configured. "
                "Please check LLM_API_URL, LLM_TOKEN, and LLM_MODEL settings."
            )

        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        self.logger.debug("Sending LLM chat request with %d messages, temperature: %.1f",
                         len(messages), temperature)

        try:
            content = self._make_api_request(payload, timeout)
            self.logger.debug("LLM chat request completed successfully")
            return content
        except Exception as e:
            self.logger.error("LLM chat request failed: %s", e)
            raise

    def call_with_prompt(self, prompt: str, system_prompt: Optional[str] = None,
                         temperature: float = 0.0, timeout: int = 30) -> str:
        """Convenience method for single-prompt LLM calls.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt (defaults to generic assistant)
            temperature: Sampling temperature
            timeout: Request timeout in seconds

        Returns:
            Generated text content from LLM
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, temperature=temperature, timeout=timeout)

    def call_with_context(self, query: str, context_markdown: Optional[str] = None,
                          response_language: str = "Chinese", temperature: float = 0.0) -> Tuple[str, Any]:
        """Call LLM with query and optional context (replaces utils.call_llm).

        Args:
            query: User query/question
            context_markdown: Optional context in markdown format
            response_language: Language for response
            temperature: Sampling temperature

        Returns:
            Tuple of (content_text, raw_data)
        """
        if not self.is_configured():
            self.logger.warning("LLM call skipped: client not configured")
            return "LLM call skipped: client not configured", None

        messages = []

        # Build system prompt with language instruction
        system_prompt = f"You are a helpful assistant. Respond in {response_language}."
        messages.append({"role": "system", "content": system_prompt})

        # Build user message with context if provided
        user_message = query
        if context_markdown:
            user_message = f"Context:\n```markdown\n{context_markdown}\n```\n\nQuestion: {query}"

        messages.append({"role": "user", "content": user_message})

        self.logger.info("Calling LLM with query: %s", query[:100])

        try:
            content = self.chat(messages, temperature=temperature, timeout=30)
            # Return content and None for raw_data (compatibility with utils.call_llm)
            return content, None
        except Exception as e:
            error_msg = f"LLM call failed: {e}"
            self.logger.error(error_msg)
            return error_msg, None

    def call_api_simple(self, query: str, context_markdown: Optional[str] = None,
                        response_language: str = "Chinese") -> str:
        """Simple wrapper that returns only text content (replaces utils.call_llm_api).

        Args:
            query: User query/question
            context_markdown: Optional context in markdown format
            response_language: Language for response

        Returns:
            Text content from LLM
        """
        content, _ = self.call_with_context(
            query=query,
            context_markdown=context_markdown,
            response_language=response_language
        )
        return content

    def generate_summary_for_chunk(self, chunk: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Generate summary and keywords for a code chunk.

        Args:
            chunk: Code chunk metadata dictionary

        Returns:
            Tuple of (summary_text, keyword_list)
        """
        if not self.is_configured():
            self.logger.warning("LLM not configured for summary generation")
            return "", []

        source_code = chunk.get("code", "")
        prompt = (
            "Analyze the following xv6 code to generate a concise summary and a few relevant keywords.\n"
            f"NAME: {chunk.get('name')}\n"
            f"TYPE: {chunk.get('type')}\n"
            f"FILE: {chunk.get('file')}\n"
            f"CODE:\n{source_code}\n\n"
            "Response ONLY in JSON format:\n"
            '{"summary": "...", "keywords": ["kw1", "kw2", ...]}'
        )

        messages = [
            {"role": "system", "content": "You are an xv6 code parser. Output only JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            content = self.chat(messages, temperature=0.1, timeout=30)
            return self._parse_summary_response(content)
        except Exception as e:
            self.logger.error("Error generating summary for chunk %s: %s", chunk.get("id"), e)
            return "", []

    def _make_api_request(self, payload: Dict[str, Any], timeout: int) -> str:
        """Make HTTP request to LLM API.

        Args:
            payload: API payload dictionary
            timeout: Request timeout in seconds

        Returns:
            Raw response content from LLM

        Raises:
            urllib.error.HTTPError: If API request fails
            json.JSONDecodeError: If response is not valid JSON
        """
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

    def _parse_summary_response(self, content: str) -> Tuple[str, List[str]]:
        """Parse LLM response to extract summary and keywords.

        Args:
            content: Raw LLM response content

        Returns:
            Tuple of (summary_text, keyword_list)
        """
        # Extract JSON from code blocks if present
        cleaned_content = self._extract_json_from_code_blocks(content)

        try:
            result = json.loads(cleaned_content)
            summary = str(result.get("summary", "")).strip()
            keywords = result.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(k).strip() for k in keywords if str(k).strip()]
            return summary, keywords
        except Exception:
            # Fallback: keep a concise plain-text summary if model output is not valid JSON.
            fallback_summary = " ".join(content.split())[:280].strip()
            return fallback_summary, []

    def _extract_json_from_code_blocks(self, text: str) -> str:
        """Extract JSON content from code blocks in LLM response.

        Args:
            text: Raw LLM response text

        Returns:
            Cleaned text with code blocks removed
        """
        if "```json" in text:
            return text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        return text