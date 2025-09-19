# src/utils/config_parser.py
import os
import yaml
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Simple functional API to load YAML config.
    Falls back to default config if file missing or invalid.
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", os.path.join("configs", "config.yaml"))

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return ConfigParser()._get_default_config()

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML: {e}. Using defaults.")
            return ConfigParser()._get_default_config()

    return config


class ConfigParser:
    """
    Advanced config parser with:
    - Default config
    - Environment variable substitution
    - Section and key access
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            self.config_path = os.environ.get(
                "CONFIG_PATH",
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             "configs", "config.yaml")
            )
        else:
            self.config_path = config_path

        self._config = None
        self._load_config()

    def _load_config(self):
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}")
                self._config = self._get_default_config()
                return

            with open(self.config_path, "r", encoding="utf-8") as file:
                raw_config = yaml.safe_load(file)

            # Substitute env vars then normalize shape/compatibility for consumers
            substituted = self._substitute_env_vars(raw_config)
            self._config = self._normalize_config(substituted)

            logger.info(f"Config loaded: {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            # Even on failure, provide normalized defaults so consumers work
            self._config = self._normalize_config(self._get_default_config())

    def _substitute_env_vars(self, config: Any) -> Any:
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(v) for v in config]
        elif isinstance(config, str):
            pattern = r"\$\{([^}]+)\}"
            return re.sub(pattern, lambda m: os.getenv(m.group(1), m.group(0)), config)
        return config

    def get_config(self) -> Dict[str, Any]:
        return self._config.copy() if self._config else {}

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {}) if self._config else {}

    def get_value(self, key_path: str, default: Any = None) -> Any:
        if not self._config:
            return default

        keys = key_path.split(".")
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and backfill configuration so existing modules can rely on
        stable keys without being changed.

        - Ensure embedding.model_name and embedding.device exist (mirror nested provider if needed)
        - Ensure retriever section exists with sensible defaults
        - Ensure text_splitter section exists in factory-friendly shape
        - Keep existing keys intact; only add missing compatibility keys
        """
        try:
            cfg = dict(config or {})

            # --- Embedding normalization ---
            embedding_cfg = cfg.get("embedding", {}) or {}
            provider = (embedding_cfg.get("provider") or "huggingface").lower()
            provider_cfg = embedding_cfg.get(provider, {}) if isinstance(embedding_cfg.get(provider), dict) else {}

            # Backfill top-level embedding keys used by QueryEmbeddingService
            if "model_name" not in embedding_cfg and "model_name" in provider_cfg:
                embedding_cfg["model_name"] = provider_cfg.get("model_name")
            if "device" not in embedding_cfg and "device" in provider_cfg:
                embedding_cfg["device"] = provider_cfg.get("device") or "auto"
            # Reasonable defaults if still missing
            embedding_cfg.setdefault("model_name", "intfloat/multilingual-e5-base")
            embedding_cfg.setdefault("device", "auto")
            cfg["embedding"] = embedding_cfg

            # --- Retriever normalization ---
            retriever_cfg = cfg.get("retriever") or {}
            retriever_cfg.setdefault("vector_weight", 0.85)
            retriever_cfg.setdefault("keyword_weight", 0.15)
            retriever_cfg.setdefault("min_word_length", 3)
            cfg["retriever"] = retriever_cfg

            # --- text_splitter normalization for chunking ---
            # DocumentChunker expects cfg["text_splitter"] shaped as
            # { "strategy": <name>, "parameters": { ... } }
            if "text_splitter" not in cfg or not isinstance(cfg.get("text_splitter"), dict):
                # Provide factory-aligned defaults matching ChunkingFactory.get_chunker defaults
                cfg["text_splitter"] = {
                    "strategy": "chained",
                    "parameters": {
                        "header_config": {
                            "max_chunk_size": 1500,
                            "chunk_overlap": 200,
                            "min_chunk_size": 100,
                            "header_font_threshold": 14.0,
                        },
                        "semantic_config": {
                            "max_chunk_size": 1200,
                            "chunk_overlap": 150,
                            "min_chunk_size": 200,
                            "sentence_min_length": 10,
                        },
                    },
                }

            # --- LLM sane defaults ---
            llm_cfg = cfg.get("llm") or {}
            llm_cfg.setdefault("provider", "huggingface")
            llm_cfg.setdefault("model_name", "meta-llama/Meta-Llama-3.1-8B-Instruct")
            llm_cfg.setdefault("temperature", 0.5)
            llm_cfg.setdefault("max_tokens", 512)
            llm_cfg.setdefault("language", "de-turkish")
            # api_token intentionally left possibly None (set via UI); consumer will validate
            cfg["llm"] = llm_cfg

            return cfg
        except Exception:
            # On any normalization error, fall back to defaults
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "embedding": {
                "provider": "huggingface",
                # Top-level keys for consumers (mirrored from provider section)
                "model_name": "intfloat/multilingual-e5-base",
                "device": "auto",
                "huggingface": {
                    "model_name": "intfloat/multilingual-e5-base",
                    "device": None,
                    "max_length": 512,
                    "batch_size": 32,
                    "normalize_embeddings": True,
                    "language": "de",
                    "text_cleaning": {
                        "remove_special_chars": True,
                        "normalize_whitespace": True,
                        "lowercase": False
                    }
                },
            },
            "vectordb": {
                "provider": "chroma",
                "chroma": {
                    "persist_directory": "./data/chroma_db",
                    "collection_name": "akademische_dokumente",
                    "settings": {
                        "anonymized_telemetry": False,
                        "allow_reset": True
                    },
                    "metadata": {
                        "description": "Sammlung akademischer Dokumente und Forschungsarbeiten",
                        "language": "de",
                        "domain": "academic",
                        "version": "1.0"
                    }
                },
            },
            "chunking": {
                "chunk_size": 512, 
                "chunk_overlap": 50, 
                "strategy": "recursive",
                "language_specific": {
                    "preserve_sentences": True,
                    "respect_paragraphs": True,
                    "handle_compound_words": True
                },
                "table_detection": {
                    "enabled": True,
                    "indicators": [
                        "Tabelle", "Tab.", "Spalte", "Zeile",
                        "siehe Tabelle", "in der Tabelle", "─", "│"
                    ]
                }
            },
            "text_cleaning": {
                "german": {
                    "remove_umlauts": False,
                    "normalize_umlauts": True,
                    "handle_compound_words": True,
                    "preserve_case": True,
                    "remove_stopwords": False,
                    "custom_stopwords": ["bzw.", "etc.", "usw.", "z.B.", "d.h."]
                }
            },
            "retrieval": {
                "top_k": 5, 
                "score_threshold": 0.0,
                "min_similarity": 0.3,
                "filters": {
                    "language": "de",
                    "content_type": ["text", "table", "figure"],
                    "academic_domain": ["economics", "finance", "business"]
                }
            },
            # Defaults for the hybrid retriever consumer
            "retriever": {
                "vector_weight": 0.85,
                "keyword_weight": 0.15,
                "min_word_length": 3
            },
            "llm": {
                "provider": "huggingface",
                "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "temperature": 0.5,
                "max_tokens": 512,
                "language": "de-turkish",
                "api_token": None,
                "system_prompt": "Du bist ein hilfreicher akademischer Assistent, der auf Deutsch antwortet. Beantworte Fragen basierend auf den bereitgestellten Dokumenten. Verwende eine formelle, akademische Sprache."
            },
            # Back-compat: provide a factory-ready text_splitter for chunker
            "text_splitter": {
                "strategy": "chained",
                "parameters": {
                    "header_config": {
                        "max_chunk_size": 1500,
                        "chunk_overlap": 200,
                        "min_chunk_size": 100,
                        "header_font_threshold": 14.0
                    },
                    "semantic_config": {
                        "max_chunk_size": 1200,
                        "chunk_overlap": 150,
                        "min_chunk_size": 200,
                        "sentence_min_length": 10
                    }
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "./logs/academic_assistant.log",
                "max_size": "10MB",
                "backup_count": 5
            },
            "data": {
                "input_directory": "./data/raw",
                "processed_directory": "./data/processed", 
                "output_directory": "./data/output",
                "supported_formats": ["pdf", "txt", "docx"]
            },
            "performance": {
                "batch_size": 100,
                "max_workers": 4,
                "cache_embeddings": True,
                "cache_directory": "./cache/embeddings"
            },
            "testing": {
                "test_data_directory": "./tests/data",
                "mock_embeddings": True,
                "test_collection_name": "test_collection",
                "cleanup_after_tests": True
            }
        }

# Global config instance
CONFIG = load_config()