# tests/test_pdf_loader.py

import pytest
from pathlib import Path
from src.ingestion.pdf_loader import PDFLoader
from fpdf import FPDF

@pytest.fixture
def sample_pdf_path(tmp_path):
    """
    Creates a temporary sample PDF for testing.
    """
    pdf_file = tmp_path / "sample.pdf"
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "This is a test PDF.\nIt has multiple lines of text.")
    pdf.add_page()
    pdf.multi_cell(0, 10, "Second page content here.")
    
    pdf.output(str(pdf_file))
    return pdf_file

def test_pdf_loader_load(sample_pdf_path):
    """
    Tests that PDFLoader.load() correctly processes PDF pages and returns required fields.
    """
    loader = PDFLoader()
    documents = loader.load(str(sample_pdf_path))

    assert documents is not None
    assert len(documents) == 2

    for doc in documents:
        assert "chunk_id" in doc
        assert "source_file" in doc
        assert "page_number" in doc
        assert "text" in doc
        assert isinstance(doc["tables"], list)
        assert isinstance(doc["words"], list)
        # Optionally check that text is not empty
        assert len(doc["text"]) > 0
