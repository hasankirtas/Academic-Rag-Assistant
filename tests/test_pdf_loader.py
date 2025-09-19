# test.test_pdf_loader.py
import pytest
from pathlib import Path
from src.ingestion.pdf_loader import PDFLoader
from fpdf import FPDF

@pytest.fixture
def sample_pdf_path(tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Dies ist ein Test-PDF.\nEs enthält mehrere Zeilen von Text.\nHier ist noch eine Zeile.")
    
    pdf.add_page()
    pdf.multi_cell(0, 10, "Dies ist die zweite Seite.\nHier steht weiterer Text für den Test.\nEnde des Beispiels.")
    
    pdf.output(str(pdf_file))
    return pdf_file

def test_pdf_loader_load(sample_pdf_path):
    loader = PDFLoader()
    documents = loader.load(str(sample_pdf_path))

    assert documents
    assert len(documents) >= 2

    for doc in documents:
        assert "chunk_id" in doc
        assert "source_file" in doc
        assert "page_number" in doc
        assert "text" in doc
        assert "element_type" in doc
        assert len(doc["text"]) > 0

    for i, doc in enumerate(documents, 1):
        print(f"Chunk {i}: page {doc['page_number']}, type {doc['element_type']}, text length {len(doc['text'])}")