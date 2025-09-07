# tests/test_text_cleaner.py
import pytest
from src.preprocessing.text_cleaner import TextCleaner
from src.core.factories.cleaner_factory import get_cleaner_strategies

@pytest.fixture
def sample_text():
    return """
    Das•BIP   (Bruttoinlandsprodukt)   ist ein   wichtiger   Indikator.  
    Konjunktur- schwankungen  beeinﬂussen   die  Wirtschaft.

    ►   Wichtige  Punkte:
    • Makroökonomische   Theorie
    ▪  Geldp0litik   der EZß
    → dasB1P wächßt   stet1g  

    Prof.Dr.Schmidt  erklärt: „Die Inflati0n   ist  niedrig." 
    (vgl. Müller, 2020,   S. 123-126).  

    Abb. 1: Wirtschaftswachstum ¦ 1990—2020 ⟶ inkorrekt OCR.
    """

@pytest.fixture
def text_cleaner():
    strategies = get_cleaner_strategies()
    return TextCleaner(strategies)

def test_clean_returns_string(text_cleaner, sample_text):
    cleaned = text_cleaner.clean(sample_text)
    assert isinstance(cleaned, str)
    assert len(cleaned.strip()) > 0

def test_preview_cleaning_structure(text_cleaner, sample_text):
    preview = text_cleaner.preview_cleaning(sample_text, max_length=200)
    assert "original" in preview
    assert "steps" in preview
    assert "final" in preview
    assert len(preview["steps"]) == len(text_cleaner.strategies)

def test_clean_batch_with_metadata(text_cleaner, sample_text):
    pages = [
        {"page_number": 1, "text": sample_text},
        {"page_number": 2, "text": "Ein kurzer Testtext."},
        {"page_number": 3, "text": ""}
    ]
    cleaned_pages = text_cleaner.clean_batch_with_metadata(pages)
    assert len(cleaned_pages) == 3
    for page in cleaned_pages:
        assert "page_number" in page
        assert "text" in page
        assert isinstance(page["text"], str)

def test_clean_raises_typeerror_on_non_string(text_cleaner):
    with pytest.raises(TypeError):
        text_cleaner.clean(123)