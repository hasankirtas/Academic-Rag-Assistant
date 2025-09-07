"""
Factory for creating chunking strategies.

This module provides factory functions to create different chunking strategies based on
configuration parameters, following the Factory pattern for flexible strategy instantiation.
"""

from typing import Dict, Any, Optional
from src.core.abstractions.chunking_strategy import LogicalChunkingStrategy
from src.implementations.chunking.logical_chunking_strategies import (
    HeaderBasedChunker,
    SemanticChunker,
    ChainedChunker
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ChunkingFactory:
    """
    Factory class for creating chunking strategy instances.
    """

    _strategies = {
        'header_based': HeaderBasedChunker,
        'semantic': SemanticChunker,
        'chained': ChainedChunker  # default strategy
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> LogicalChunkingStrategy:
        if strategy_name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(f"Unknown chunking strategy: {strategy_name}. Available: {available}")

        strategy_class = cls._strategies[strategy_name]

        # Create instance
        if strategy_name == 'chained':
            # ChainedChunker expects separate header and semantic config
            header_config = config.get('header_config', {})
            semantic_config = config.get('semantic_config', {})
            instance = strategy_class(header_config=header_config, semantic_config=semantic_config)
        else:
            instance = strategy_class(config)

        # Validate configuration parameters
        if hasattr(instance, 'get_config_requirements'):
            required_params = instance.get_config_requirements()
            missing_params = [param for param in required_params if param not in (config if strategy_name != 'chained' else config.get('parameters', {}))]
            if missing_params:
                logger.error(f"Missing configuration parameters for {strategy_name}: {missing_params}")
                raise KeyError(f"Missing required configuration parameters: {missing_params}")

        logger.debug(f"Creating {strategy_name} chunking strategy with config: {config}")
        return instance

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        if not issubclass(strategy_class, LogicalChunkingStrategy):
            raise TypeError(f"Strategy class must implement LogicalChunkingStrategy")

        cls._strategies[name] = strategy_class
        logger.info(f"Registered new chunking strategy: {name}")

    @classmethod
    def get_chunker(cls, strategy_name: str = 'chained', config: Optional[Dict[str, Any]] = None) -> LogicalChunkingStrategy:
        if config is None:
            default_configs = {
                'header_based': {
                    'max_chunk_size': 1500,
                    'chunk_overlap': 200,
                    'min_chunk_size': 100,
                    'header_font_threshold': 14.0
                },
                'semantic': {
                    'max_chunk_size': 1200,
                    'chunk_overlap': 150,
                    'min_chunk_size': 200,
                    'sentence_min_length': 10
                },
                'chained': {
                    'header_config': {
                        'max_chunk_size': 1500,
                        'chunk_overlap': 200,
                        'min_chunk_size': 100,
                        'header_font_threshold': 14.0
                    },
                    'semantic_config': {
                        'max_chunk_size': 1200,
                        'chunk_overlap': 150,
                        'min_chunk_size': 200,
                        'sentence_min_length': 10
                    }
                }
            }
            config = default_configs.get(strategy_name, {})

        return cls.create_strategy(strategy_name, config)

    @classmethod
    def _create_chunker_from_config(cls, chunking_config: Dict[str, Any]) -> LogicalChunkingStrategy:
        if 'strategy' not in chunking_config:
            raise KeyError("'strategy' key is missing from configuration")

        strategy_name = chunking_config['strategy']
        params = chunking_config.get('parameters', {})

        return cls.create_strategy(strategy_name, params)

# --- Module-level convenience wrappers expected by callers ---

def get_chunker(strategy_name: str = 'chained', config: Optional[Dict[str, Any]] = None) -> LogicalChunkingStrategy:
    """
    Convenience wrapper to obtain a chunker instance by name with optional config.
    """
    return ChunkingFactory.get_chunker(strategy_name=strategy_name, config=config)


def create_chunker_from_config(chunking_config: Dict[str, Any]) -> LogicalChunkingStrategy:
    """
    Create a chunker instance from a configuration dict shaped like:
      { 'strategy': '<name>', 'parameters': { ... } }
    For the 'chained' strategy, parameters may contain 'header_config' and 'semantic_config'.
    """
    return ChunkingFactory._create_chunker_from_config(chunking_config)


def validate_chunking_config(chunking_config: Dict[str, Any]) -> None:
    """
    Validate that the provided chunking configuration is well-formed and compatible
    with the chosen strategy. Raises on validation errors.
    """
    if not isinstance(chunking_config, dict):
        raise TypeError("chunking_config must be a dict")

    if 'strategy' not in chunking_config:
        raise KeyError("chunking_config missing required key 'strategy'")

    strategy_name = chunking_config['strategy']
    available = ChunkingFactory.get_available_strategies()
    if strategy_name not in available:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {', '.join(available)}")

    params = chunking_config.get('parameters', {})
    if not isinstance(params, dict):
        raise TypeError("chunking_config['parameters'] must be a dict")

    # Validate parameter requirements per strategy
    if strategy_name == 'chained':
        header_cfg = params.get('header_config', {})
        semantic_cfg = params.get('semantic_config', {})
        if not isinstance(header_cfg, dict) or not isinstance(semantic_cfg, dict):
            raise TypeError("For 'chained' strategy, 'header_config' and 'semantic_config' must be dicts")

        # Instantiate temporary strategies to read their requirements
        header_tmp = HeaderBasedChunker({})
        semantic_tmp = SemanticChunker({})
        missing_header = [k for k in header_tmp.get_config_requirements() if k not in header_cfg]
        missing_semantic = [k for k in semantic_tmp.get_config_requirements() if k not in semantic_cfg]
        if missing_header or missing_semantic:
            problems = []
            if missing_header:
                problems.append(f"header_config missing: {missing_header}")
            if missing_semantic:
                problems.append(f"semantic_config missing: {missing_semantic}")
            raise KeyError("; ".join(problems))
        return

    # Non-chained strategies
    # Create a temporary instance to query requirements without enforcing behavior
    if strategy_name == 'header_based':
        tmp = HeaderBasedChunker({})
    elif strategy_name == 'semantic':
        tmp = SemanticChunker({})
    else:
        # Fallback: try constructing via factory; if it fails, surface error
        ChunkingFactory.create_strategy(strategy_name, params)
        return

    required = tmp.get_config_requirements()
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"Missing required configuration parameters for '{strategy_name}': {missing}")