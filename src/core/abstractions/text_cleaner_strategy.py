"""
Text cleaning strategy abstraction for German Academic PDF content processing.
"""

from abc import ABC, abstractmethod

class TextCleanerStrategy(ABC):
    """
    Abstract base class for text cleaning strategies

    This allows for flexible composition or different cleaning operations 
    while maintaining the Strategy pattern for easy testing and extension
    """

    @abstractmethod
    def clean(self, text:str) -> str:
        """
        Clean the input text according to the specific strategy.

        Args:
            text (str): Raw text to be cleaned

        Returns:
            str: Cleaned Text
        """
        raise NotImplementedError("Subclasses must implement the clean() method")

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the cleaning strategy for logging purposes.

        Returns:
            str: Strategy Name
        """
        raise NotImplementedError("Subclasses must implement the name property")    