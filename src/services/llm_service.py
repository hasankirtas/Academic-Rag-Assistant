"""
LLM Service for RAG System (OpenAI GPT-5-mini)

Provides text generation using OpenAI API (GPT-5-mini).
"""

from typing import List, Dict, Any, Optional, Generator
import time
import logging
import os
import sys
import unicodedata
from openai import OpenAI
import src.utils.config_parser as config_module

from src.utils.logger import setup_logger
from src.utils.config_parser import CONFIG
from src.utils.response_cache import get_response_cache

logger = setup_logger(__name__)


# Ensure UTF-8 runtime to avoid ASCII codec errors in some Windows environments
try:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "en_US.UTF-8")
    os.environ.setdefault("LC_ALL", "en_US.UTF-8")
    # Disable Chroma telemetry to avoid telemetry exceptions in some installs
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _ensure_utf8(text: str) -> str:
    """Normalize and ensure a safe UTF-8 encodable string."""
    if not isinstance(text, str):
        text = str(text)
    # NFC normalization helps with composed characters
    normalized = unicodedata.normalize("NFC", text)
    # Round-trip encode/decode to guarantee utf-8 safety without raising
    return normalized.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


class LLMService:
    """
    Service for text generation using OpenAI API (GPT-5-mini).
    """

    def __init__(self, model_name: Optional[str] = None):
        # Access CONFIG dynamically so test patches are honored
        llm_config = config_module.CONFIG.get("llm", {})

        self.model_name = model_name or llm_config.get(
            "model_name", "gpt-5-mini"
        )
        # GPT-5-mini only supports temperature=1 (default), so we don't use this parameter
        self.temperature = llm_config.get("temperature", 1.0)  # Not used in API calls
        self.max_tokens = llm_config.get("max_tokens", 512)
        self.language = llm_config.get("language", "de")

        # Get API key from config or environment
        api_token = llm_config.get("api_token", None)
        if not api_token:
            api_token = os.getenv("OPENAI_API_KEY")
        
        if not api_token:
            raise ValueError("OpenAI API Key is missing. Please set OPENAI_API_KEY environment variable or configure in config.yaml")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_token)
        self.cache = get_response_cache()
        logger.info(f"Initialized OpenAI LLM service with {self.model_name}")

    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Build a concise instruction prompt for German academic content.
        
        Args:
            query: User query
            contexts: List of context dictionaries with 'text' and 'hybrid_score'
        
        Returns:
            Formatted instruction prompt string
        """

        # 1. Prepare context string (limit to top 5 most relevant)
        context_str = ""
        if contexts:
            sorted_contexts = sorted(contexts, key=lambda x: x.get('hybrid_score', 0), reverse=True)[:5]
            for i, ctx in enumerate(sorted_contexts, 1):
                text = ctx.get("text", "")[:1000]  # Shorter context chunks
                context_str += f"{i}. {text}\n\n"

        # 2. Concise instruction template
        base_prompt = f"""Du bist ein akademischer Assistent. Beantworte die Frage basierend auf den Dokumentenauszügen.

Kontext:
{context_str if context_str else "Keine Dokumente verfügbar."}

Regeln:
- Antworte präzise und akademisch
- Nutze nur die bereitgestellten Dokumente
- Bei Türkisch: Antworte auf Türkisch mit deutschen Fachbegriffen
- Bei Deutsch: Antworte auf Deutsch
- Maximal 300 Wörter

Frage: {query}

Antwort:"""

        return base_prompt.strip()

    # Public wrapper to satisfy external callers that expect a build_prompt API
    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        return self._build_prompt(query, contexts)

    def _call_openai_api(self, prompt: str, stream: bool = False):
        """
        Unified method to call OpenAI API with proper error handling.
        
        Args:
            prompt: The formatted prompt to send to the model
            stream: Whether to stream the response
            
        Returns:
            OpenAI API response or stream generator
        """
        try:
            # Ensure UTF-8 safe payload
            safe_prompt = _ensure_utf8(prompt)
            messages = [
                {"role": "system", "content": _ensure_utf8("Du bist ein hilfreicher akademischer Assistent, der auf Deutsch antwortet. Beantworte Fragen basierend auf den bereitgestellten Dokumenten. Verwende eine formelle, akademische Sprache.")},
                {"role": "user", "content": safe_prompt}
            ]
            
            # GPT-5-mini only supports temperature=1 (default)
            # Remove temperature parameter to use default value
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_tokens,
                stream=stream,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            if stream:
                return iter([])  # Return empty generator for streaming
            else:
                raise e

    def generate_response(self, query: str, contexts: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate a complete response for the given query and contexts with caching.
        
        Args:
            query: User query
            contexts: List of context dictionaries with 'text' and 'hybrid_score'
            
        Returns:
            Dictionary containing answer, metadata, and timing information
        """
        contexts = contexts or []
        
        # Check cache first
        cached_response = self.cache.get(query, contexts, self.model_name)
        if cached_response is not None:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached_response
        
        start = time.time()
        prompt = self._build_prompt(query, contexts)
        prompt = _ensure_utf8(prompt)
        
        try:
            response = self._call_openai_api(prompt, stream=False)
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            answer = "Entschuldigung, es gab einen Fehler bei der Antwortgenerierung."
        
        elapsed = (time.time() - start) * 1000

        result = {
            "answer": answer,
            "prompt": prompt,
            "generation_time_ms": elapsed,
            "model_name": self.model_name,
            "contexts_used": len(contexts),
            "cached": False
        }
        
        # Cache the result
        self.cache.put(query, contexts, self.model_name, result)
        logger.debug(f"Cached response for query: {query[:50]}...")
        
        return result

    def stream_response(self, query: str, contexts: List[Dict] = None) -> Generator[str, None, None]:
        """
        Stream response tokens for the given query and contexts.
        
        Args:
            query: User query
            contexts: List of context dictionaries with 'text' and 'hybrid_score'
            
        Yields:
            String chunks of the response as they arrive
        """
        prompt = self._build_prompt(query, contexts or [])
        
        try:
            stream = self._call_openai_api(prompt, stream=True)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield _ensure_utf8(chunk.choices[0].delta.content)
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            yield "Entschuldigung, es gab einen Fehler bei der Antwortgenerierung."

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "language": self.language,
        }


def create_llm_service(model_name: Optional[str] = None) -> LLMService:
    service = LLMService(model_name=model_name)
    logger.info("LLMService created successfully")
    return service


if __name__ == "__main__":
    llm = create_llm_service()
    query = "Was ist das Bruttoinlandsprodukt?"
    contexts = [
        {"text": "Das Bruttoinlandsprodukt misst den Wert aller Güter und Dienstleistungen.", "hybrid_score": 0.85}
    ]

    response = llm.generate_response(query, contexts)
    print("Answer:", response["answer"])
    print("Generation time:", response["generation_time_ms"], "ms")
    print("Model info:", llm.get_model_info())