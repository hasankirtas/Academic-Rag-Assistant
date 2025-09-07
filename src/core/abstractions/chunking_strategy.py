from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LogicalChunkingStrategy(ABC):
    """
    Abstract base class for logical chunking strategies.

    Defines the contract for document segmentation based on logical
    or semantic boundaries. Simplified to focus on text and essential metadata.
    """

    @abstractmethod
    def chunk(self, structured_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk the structured document data into logical segments.

        Args:
            structured_data (List[Dict[str, Any]]): List of structured elements
                from PDF processing. Each dict contains:
                - 'text': str - The text content
                - 'page_number': int - Page number where text appears
                - Optional metadata like 'tables', 'words', 'source_file', 'chunk_id'

        Returns:
            List[Dict[str, Any]]: List of chunks, each containing:
                - 'content': str - The chunked text content
                - 'metadata': Dict - Metadata about the chunk including:
                    - 'page_number': int - Primary page number
                    - 'page_range': tuple - (start_page, end_page)
                    - 'section_title': str - Section or chapter title (if any)
                    - 'chunk_index': int - Index of chunk in document
                    - 'contains_table': bool - Whether chunk contains table
                    - 'word_count': int - Number of words in chunk
                    - 'char_count': int - Number of characters in chunk
        """
        raise NotImplementedError("Subclasses must implement 'chunk' method")

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        Get the name of the chunking strategy.
        
        Returns:
            str: Name of the strategy for identification and logging
        """
        raise NotImplementedError("Subclasses must implement 'strategy_name' property")


    @abstractmethod
    def get_config_requirements(self) -> List[str]:
        """
        Get the list of configuration parameters required by this strategy.
        
        Returns:
            List[str]: List of config parameter names needed
        """
        raise NotImplementedError("Subclasses must implement 'get_config_requirements' method")
