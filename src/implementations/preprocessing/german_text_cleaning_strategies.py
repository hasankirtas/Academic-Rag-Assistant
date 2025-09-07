"""
German academic text cleaning strategies for PDF content processing.
"""

import re
from typing import Dict, Text
from src.core.abstractions.text_cleaner_strategy import TextCleanerStrategy


class GermanTermPreserver(TextCleanerStrategy):
    """
    Preserves German academic and technical terms during cleaning.
    """

    def __init__(self):
        self.protected_terms = {
            # Economic terms
            "BIP", "Konjunkturschwankungen", "Volkswirtschaftslehre", "Makroökonomie",
            "Mikroökonomie", "Geldpolitik", "Fiskalpolitik", "Arbeitslosigkeit",
            "Inflation", "Deflation", "Rezession", "Wachstum", "Produktivität",
            
            # Technical terms
            "bzw.", "z.B.", "d.h.", "u.a.", "usw.", "etc.", "ca.", "Nr.",
            "Abs.", "Art.", "Hrsg.", "Aufl.", "S.", "Bd.", "Jg.", "H.",
            
            # Academic abbreviations
            "Prof.", "Dr.", "habil.", "M.A.", "B.A.", "Ph.D.", "Dipl."
        }

    @property
    def name(self) -> str:
        return "GermanTermPreserver"

    def clean(self, text: str) -> str:
        """
        Mark protected terms to prevent modification in later cleaning steps.
        Embed the original term into the placeholder so it can be restored
        without external shared state.
        """
        protected_text = text

        # Create embedded placeholders for protected terms
        # Format: __PROTECTED_TERM_{i}__{TERM}__
        for i, term in enumerate(self.protected_terms):
            placeholder = f"__PROTECTED_TERM_{i}__{term}__"
            if term in protected_text:
                protected_text = protected_text.replace(term, placeholder)

        return protected_text


class ControlCharacterCleaner(TextCleanerStrategy):
    """
    Cleans control characters and normalizes line breaks.
    Also handles common PDF artifacts like soft-hyphens and ligatures.
    """

    def __init__(self) -> None:
        # Common Latin ligatures seen in PDFs
        self._ligatures = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
        }

    @property
    def name(self) -> str:
        return "ControlCharacterCleaner"

    def clean(self, text: str) -> str:
        """
        Replace LS/PS characters, normalize line breaks, remove soft hyphens,
        fix hyphenation across wrapped lines, and normalize ligatures.
        """
        # Replace Line Separator (U+2028) and Paragraph Separator (U+2029)
        text = text.replace('\u2028', '\n')
        text = text.replace('\u2029', '\n')
        
        # Replace other problematic Unicode characters
        text = text.replace('\u00A0', ' ')  # Non-breaking space
        text = text.replace('\u200B', '')   # Zero-width space
        text = text.replace('\u200C', '')   # Zero-width non-joiner
        text = text.replace('\u200D', '')   # Zero-width joiner
        text = text.replace('\uFEFF', '')   # Byte order mark
        text = text.replace('\u00AD', '')   # Soft hyphen

        # Normalize Latin ligatures
        for lig, plain in self._ligatures.items():
            if lig in text:
                text = text.replace(lig, plain)

        # Join words that were split with hyphen + newline: "Wirt-\nschaft" -> "Wirtschaft"
        text = re.sub(r"(\w)[-\u00AD]\n(\w)", r"\1\2", text)
        # Also handle hyphenation with spaces around newline
        text = re.sub(r"(\w)[-\u00AD]\s*\n\s*(\w)", r"\1\2", text)

        return text


class WhitespaceNormalizer(TextCleanerStrategy):
    """
    Normalizes whitespace and removes excessive spacing.
    """

    @property
    def name(self) -> str:
        return "WhitespaceNormalizer"

    def clean(self, text: str) -> str:
        """
        Clean excessive whitespace while preserving paragraph structure.
        """
        # Replace multiple spaces with single space (but not across newlines)
        text = re.sub(r"(?!\n) {2,}", " ", text)
        
        # Replace multiple tabs with single space
        text = re.sub(r"\t+", " ", text)
        
        # Normalize line breaks (max 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Trim trailing whitespace from lines
        text = re.sub(r"[ \t]+\n", "\n", text)
        
        # Space normalization around punctuation: " ," -> "," and ".A" -> ". A"
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,.;:!?])(\S)", r"\1 \2", text)

        # Conservatively remove standalone page-number lines (1-4 digits)
        text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


class BulletPointStandardizer(TextCleanerStrategy):
    """
    Standardizes bullet points and list formatting.
    """

    @property
    def name(self) -> str:
        return "BulletPointStandardizer"
    
    def clean(self, text: str) -> str:
        """
        Standardize various bullet point formats to '- '. Also normalize
        numbered lists like "1) item" or "1. item" to a consistent form.
        """
        # Common bullet point characters
        bullet_patterns = [
            r"•", r"◦", r"▪", r"▫", r"‣", r"⁃",
            r"→", r"➤", r"➢", r"➣", r"►",
            r"○", r"●", r"◯", r"◉",
        ]
        
        for pattern in bullet_patterns:
            # Replace bullet at start of line
            text = re.sub(f"^{pattern}\\s*", "- ", text, flags=re.MULTILINE)
            # Replace bullet after whitespace
            text = re.sub(f"\\s+{pattern}\\s*", "\n- ", text)

        # Normalize numbered lists at line start:  "1) text" / "1 - text" / "1.  text" -> "1. text"
        text = re.sub(r"^(\s*)(\d+)[\)\.-]?\s+", r"\1\2. ", text, flags=re.MULTILINE)
        
        return text


class OCRErrorCorrector(TextCleanerStrategy):
    """
    Corrects common OCR errors in German academic texts.
    """

    def __init__(self):
        self.ocr_corrections = {
            # Common OCR word joining errors - keep word boundaries
            r"\bdasBIP\b": "das BIP",
            r"\bderEU\b": "der EU",
            r"\bimJahr\b": "im Jahr",
            r"\bzudem\b": "zu dem",
            r"\baufgrund\b": "auf Grund",

            # Common character misrecognitions
            r"0(?=[a-zA-Z])": "o",  # 0 before letters should be o
            r"(?<=[a-zA-Z])0": "o",  # 0 after letters should be o

            # rn -> m using word boundaries instead of variable-length look-behind
            r"\brn\b": "m",

            # German specific corrections
            r"ß(?=\s[A-Z])": "ss",  # ß before capital letters
            r"(?<=\w)1(?=\w)": "l",  # 1 between letters should be l
            r"(?<=\w)I(?=[a-z])": "l", # Capital I before lowercase should be l
        }

    @property
    def name(self) -> str:
        return "OCRErrorCorrector"

    def clean(self, text: str) -> str:
        """
        Apply OCR error corrections.
        """
        corrected_text = text

        for pattern, replacement in self.ocr_corrections.items():
            corrected_text = re.sub(pattern, replacement, corrected_text)

        return corrected_text



class SymbolCleaner(TextCleanerStrategy):
    """
    Removes or normalizes unwanted symbols and characters.
    """

    @property
    def name(self) -> str:
        return "SymbolCleaner"
    
    def clean(self, text: str) -> str:
        """
        Clean unwanted symbols while preserving meaningful punctuation.
        """
        # Normalize quotes and apostrophes
        text = re.sub(r"[‚„“”\"]", '"', text)
        text = re.sub(r"[’'`]", "'", text)
        
        # Normalize dashes and ellipsis
        text = re.sub(r"[–—]", "-", text)
        text = re.sub(r"…", "...", text)
        
        # Remove excessive punctuation (but keep academic notation)
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r",\s*,+", ",", text)
        
        # Remove standalone symbols that don't add meaning
        text = re.sub(r"^\s*[►▪▫•◦‣⁃]\s*$", "", text, flags=re.MULTILINE)
        
        return text


class TermRestorer(TextCleanerStrategy):
    """
    Restores protected German academic terms after other cleaning operations.
    """
    
    @property
    def name(self) -> str:
        return "TermRestorer"
    
    def clean(self, text: str) -> str:
        """
        Restore protected terms that were temporarily replaced.
        Decodes embedded placeholders produced by GermanTermPreserver.
        """
        restored_text = text
        # Replace placeholders of the form __PROTECTED_TERM_{i}__{TERM}__ with {TERM}
        restored_text = re.sub(r"__PROTECTED_TERM_\d+__(.*?)__", lambda m: m.group(1), restored_text)
        return restored_text