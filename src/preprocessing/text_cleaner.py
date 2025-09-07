from typing import List, Optional, Dict
import logging
from src.core.factories.cleaner_factory import get_cleaner_strategies
from src.core.abstractions.text_cleaner_strategy import TextCleanerStrategy
from src.utils.logger import get_logger


class TextCleaner:
    """
    Service class for cleaning raw text using cleaning strategies, preserving
    metadata for each chunk/page.
    """

    def __init__(
        self,
        strategies: Optional[List[TextCleanerStrategy]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the TextCleaner with a list of cleaning strategies.
        """
        try:
            self.strategies = strategies or get_cleaner_strategies()
        except Exception as e:
            raise RuntimeError(f"Failed to load cleaner strategies: {str(e)}")

        self.logger = logger or get_logger(__name__)
        strategy_names = [strategy.name for strategy in self.strategies]
        self.logger.info(f"TextCleaner initialized with strategies: {strategy_names}")

    def clean(self, text: str) -> str:
        """Clean single string text."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        if not text.strip():
            self.logger.warning("Input text is empty or contains only whitespace")
            return ""

        cleaned_text = text
        for strategy in self.strategies:
            try:
                cleaned_text = strategy.clean(cleaned_text)
            except Exception as e:
                self.logger.error(f"Strategy '{strategy.name}' failed: {str(e)}")
        return cleaned_text

    def clean_with_metadata(self, page_data: Dict) -> Dict:
        """Clean text while preserving metadata."""
        if "text" not in page_data:
            raise ValueError("Input dictionary must contain 'text' key")
        page_data["text"] = self.clean(page_data["text"])
        return page_data

    def clean_batch_with_metadata(self, pages_data: List[Dict]) -> List[Dict]:
        """Clean multiple pages/chunks while preserving metadata."""
        cleaned_pages = []
        for page_data in pages_data:
            try:
                cleaned_pages.append(self.clean_with_metadata(page_data))
            except Exception as e:
                self.logger.error(
                    f"Failed to clean page {page_data.get('page_number', '?')}: {str(e)}"
                )
                page_data["text"] = ""
                cleaned_pages.append(page_data)
        return cleaned_pages

    def preview_cleaning(self, text: str, max_length: int = 500) -> dict:
        """
        Preview the cleaning process step by step for debugging purposes.
        """
        if not isinstance(text, str):
            return {"error": "Invalid input text"}

        preview_text = text[:max_length] + ("..." if len(text) > max_length else "")
        results = {"original": preview_text, "steps": [], "final": ""}

        cleaned_text = text
        for strategy in self.strategies:
            try:
                before_length = len(cleaned_text)
                cleaned_text = strategy.clean(cleaned_text)
                after_length = len(cleaned_text)

                preview_cleaned = (
                    cleaned_text[:max_length]
                    + ("..." if len(cleaned_text) > max_length else "")
                )

                results["steps"].append({
                    "strategy": strategy.name,
                    "before_length": before_length,
                    "after_length": after_length,
                    "preview": preview_cleaned
                })
            except Exception as e:
                self.logger.error(f"Preview failed at strategy '{strategy.name}': {str(e)}")
                results["steps"].append({
                    "strategy": strategy.name,
                    "error": str(e)
                })
                break

        results["final"] = (
            cleaned_text[:max_length] + ("..." if len(cleaned_text) > max_length else "")
        )
        return results

# Example usage and testing
if __name__ == "__main__":
    from src.core.factories.cleaner_factory import get_cleaner_strategies
    
    sample_text = """
    Das•BIP   (Bruttoinlandsprodukt)   ist ein   wichtiger   Indikator.  
    Konjunktur- schwankungen  beeinﬂussen   die  Wirtschaft.

    ►   Wichtige  Punkte:
    • Makroökonomische   Theorie
    ▪  Geldp0litik   der EZß
    → dasB1P wächßt   stet1g  

    Prof.Dr.Schmidt  erklärt: „Die Inflati0n   ist  niedrig." 
    (vgl. Müller, 2020,   S. 123-126).  

    Abb. 1: Wirtschaftswachstum ¦ 1990—2020 ⟶ inkorrekt OCR.

    ————————————————————————————————
    Dieses  Dokument enthält   zahlreiche    Sonderzeichen… 
    •∑∆≈ç√∫   und   verschiedene    Unicode-Probleme‼
    
    Fußnote: 1  Dies  ist   ein  typisches Beispiel   für   OCR-Fehler  
    in   wissenschaftlichen PDF-Dokumenten.  
    """
    
    print("Original text:")
    print(repr(sample_text))
    print("\n" + "="*50 + "\n")
    
    strategies = get_cleaner_strategies()
    cleaner = TextCleaner(strategies)
    
    cleaned = cleaner.clean(sample_text)
    print("Cleaned text:")
    print(repr(cleaned))
    print("\n" + "="*50 + "\n")
    
    preview = cleaner.preview_cleaning(sample_text, max_length=200)
    print("Step-by-step preview:")
    for step in preview["steps"]:
        if "error" in step:
            print(f"{step['strategy']}: ERROR - {step['error']}")
        else:
            print(f"{step['strategy']}: {step['before_length']} → {step['after_length']} chars")
            print(f"  Preview: {repr(step['preview'])}")
            print()
    print("="*50 + "\n")
    
    pages_data = [
        {"page_number": 1, "text": sample_text},
        {"page_number": 2, "text": "►Ein weiterer Abschnitt mit Fehlern und Sonderzeichen…"},
        {"page_number": 3, "text": ""}
    ]
    
    cleaned_batch = cleaner.clean_batch_with_metadata(pages_data)
    
    print("Batch cleaning results with metadata:")
    for page in cleaned_batch:
        print(f"Page {page['page_number']}: {repr(page['text'])}")
