"""
Embedding Factory Module

This module creates and returns the appropriate embedding strategy
based on the settings in the config file. Implements the Factory Pattern.
"""

from typing import Optional, Dict, Any
import logging

from src.implementations import embedding

from ..abstractions.embedding_strategy import EmbeddingStrategy
from ...implementations.embedding.huggingface_embedder import HuggingFaceEmbedder
from ...utils.config_parser import ConfigParser
from ...utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingFactory:
    """
    Factory class for creating embedding strategies.

    Instantiates the appropriate concrete strategy based on the
    'embedding.provider' setting in the config file.
    """

    # canonical providers (can be extended)
    SUPPORTED_PROVIDERS = {
        "huggingface": HuggingFaceEmbedder
    }

    # Alias mappings
    PROVIDER_ALIASES = {
        "hf": "huggingface"
    }

    DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Factory constructor.

        Args:
            config_path: Path to config file (default config is used if None)
            config_dict: Direct dict config (if provided, config_path is ignored)
        """

        if config_dict is not None:
            self.config = config_dict
        else:
            self.config_parser = ConfigParser(config_path)
            self.config = self.config_parser.get_config()

    def _normalize_provider(self, provider: str) -> str:
        """
        Normalize provider aliases.
        """
        provider = provider.lower().strip()
        return self.PROVIDER_ALIASES.get(provider, provider)

    def get_embedding_strategy(self, provider: Optional[str] = None, **kwargs) -> EmbeddingStrategy:
        """
        Create an embedding strategy for the given provider.

        Args:
            provider: Embedding provider to use (if None, taken from config)
            **kwargs: Additional provider-specific parameters

        Returns:
            Concrete EmbeddingStrategy implementation
        """

        # Determine provider
        if provider is None:
            try:
                provider = self.config["embedding"]["provider"]
                logger.info(f"Provider taken from config: {provider}")
            except Exception:
                logger.warning("No embedding.provider found in config, using 'huggingface' as default")
                provider = "huggingface"

        provider = self._normalize_provider(provider)

        # Provider validation
        if provider not in self.SUPPORTED_PROVIDERS:
            supported = ", ".join(self.SUPPORTED_PROVIDERS.keys())
            raise RuntimeError(
                f"Unsupported embedding provider: '{provider}'. Supported providers: {supported}"
            )

        # Get provider-specific settings from config
        embedding_config = self.config.get("embedding", {})

        provider_config = (
            embedding_config.get(provider)
            or embedding_config.get(self.PROVIDER_ALIASES.get(provider, ""), {})
            or {}
            )

        # Override with kwargs (immutable)
        provider_config = {**provider_config, **kwargs}

        # Default model fallback for HuggingFace
        if provider == "huggingface" and "model_name" not in provider_config:
            logger.warning(
                f"No 'model_name' found for HuggingFace, using default model: {self.DEFAULT_HF_MODEL}"
            )
            provider_config["model_name"] = self.DEFAULT_HF_MODEL

        # Instantiate strategy
        strategy_class = self.SUPPORTED_PROVIDERS[provider]

        try:
            strategy = strategy_class(**provider_config)
            logger.info(f"{strategy_class.__name__} strategy successfully created")
            return strategy
        except Exception as e:
            logger.error(f"Error creating {strategy_class.__name__}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error creating embedding strategy: {str(e)}") from e


    def get_available_providers(self) -> List[str]:
        """
        Return the list of supported providers (canonical names).
        """
        return list(self.SUPPORTED_PROVIDERS.keys())

    def validate_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate embedding config and report missing settings.
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "provider": None,
            "config": {},
        }

        if provider is None:
            if "embedding" not in self.config or "provider" not in self.config["embedding"]:
                validation_result["errors"].append("No 'embedding.provider' defined in config")
                validation_result["valid"] = False
                return validation_result
            provider = self.config["embedding"]["provider"]

        provider = self._normalize_provider(provider)
        validation_result["provider"] = provider

        if provider not in self.SUPPORTED_PROVIDERS:
            supported = ", ".join(self.SUPPORTED_PROVIDERS.keys())
            validation_result["errors"].append(
                f"Unsupported provider: '{provider}'. Supported: {supported}"
            )
            validation_result["valid"] = False
            return validation_result

        embedding_config = self.config.get("embedding", {})
        provider_config = (
            embedding_config.get(provider)
            or embedding_config.get(self.PROVIDER_ALIASES.get(provider, ""), {})
            or {}
        )
        validation_result["config"] = provider_config

        if provider == "huggingface" and "model_name" not in provider_config:
            validation_result["warnings"].append(
                f"No 'model_name' defined for HuggingFace, default model will be used ({self.DEFAULT_HF_MODEL})"
            )

        return validation_result

    
# Utility functions
def create_embedding_strategy(
    provider: str,
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> EmbeddingStrategy:
    """Directly create an embedding strategy with given provider and config."""
    factory = EmbeddingFactory(config_path, config_dict)
    return factory.get_embedding_strategy(provider, **kwargs)


def get_default_embedding_strategy(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> EmbeddingStrategy:
    """Return the default embedding strategy from config (or fallback)."""
    factory = EmbeddingFactory(config_path, config_dict)
    return factory.get_embedding_strategy()