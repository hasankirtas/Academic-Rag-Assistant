"""
Factory for creating text cleaning strategies.
"""

from typing import List
from src.core.abstractions.text_cleaner_strategy import TextCleanerStrategy
from src.implementations.preprocessing.german_text_cleaning_strategies import (
    GermanTermPreserver,
    ControlCharacterCleaner,
    WhitespaceNormalizer,
    BulletPointStandardizer,
    OCRErrorCorrector,
    SymbolCleaner,
    TermRestorer
)

def get_cleaner_strategies() -> List[TextCleanerStrategy]:
    """
    Factory function to create the German academic cleaning strategy list.
    
    Returns:
        List[TextCleanerStrategy]: Ordered list of cleaning strategies for German academic PDFs
    """
    return [
        GermanTermPreserver(),
        ControlCharacterCleaner(),
        WhitespaceNormalizer(),
        BulletPointStandardizer(),
        OCRErrorCorrector(),
        SymbolCleaner(),
        TermRestorer()
    ]