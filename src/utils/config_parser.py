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

            # Substitute env vars
            self._config = self._substitute_env_vars(raw_config)

            logger.info(f"Config loaded: {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self._config = self._get_default_config()

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

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "embedding": {
                "provider": "huggingface",
                "huggingface": {
                    "model_name": "intfloat/multilingual-e5-large",
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
            "llm": {
                "provider": "ollama", 
                "model_name": "llama2", 
                "temperature": 0.1, 
                "max_tokens": 512,
                "language": "de",
                "system_prompt": "Du bist ein hilfreicher akademischer Assistent, der auf Deutsch antwortet. Beantworte Fragen basierend auf den bereitgestellten Dokumenten. Verwende eine formelle, akademische Sprache."
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