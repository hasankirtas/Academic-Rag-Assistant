from turtle import pd
import pdfplumber
from typing import List, Dict, Optional
from pathlib import Path
from src.utils.logger import setup_logger
import uuid

logger = setup_logger(__name__)

class PDFLoader:
    """
    Extracts text and structural information (tables, headind possibilities) from PDF.
    """
    
    def __init__(self):
        self.pdf_path = None

    def load(self, pdf_path: str) -> List[Dict]:
        """
        Loads the PDF and processes each page.
        
        Returns:
            List[Dict]: A dictionary for each page, containing:
            - 'chunk_id': Unique ID
            - 'source_file': PDF file name
            - 'page_number': Page number
            - 'text': Raw text of the page
            - 'tables': Textual representation of tables on the page (list)
            - 'words': List of words extracted by pdfplumber (for font information)

        """

        self.pdf_path = pdf_path
        documents = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.info(f"Processing page {page_num}...")

                    raw_text = page.extract_text() or ""

                    tables = self._extract_tables_from_page(page)

                    words = page.extract_words(extra_attrs=["size", "fontname"])

                    page_data = {
                        "chunk_id": str(uuid.uuid4()),
                        "source_file": Path(pdf_path).name,
                        "page_number": page_num,
                        "text": raw_text,
                        "tables": tables,
                        "words": words
                    }
                    documents.append(page_data)

        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            raise e

        return documents

    def _extract_tables_from_page(self, page) -> List[str]:
        """
        Extract the tables on a page and convert them to text.
        """

        tables = []

        try:
            extracted_tables = page.extract_tables()
            if extracted_tables:
                for table in extracted_tables:
                    table_text = ""
                    for row in table:
                        table_text += "\t".join(str(cell) if cell is not None else "" for cell in row) + "\n"
                        tables.append(table_text.strip())
        
        except Exception as e:
            logger.warning(f"Could not extract tables from page: {e}")
        return tables

# test
if __name__ == "__main__":
    loader = PDFLoader()
    documents = loader.load("data/raw/Makrooekonomie.pdf")

    first_page = documents[0]
    print(f"Chunk ID: {first_page['chunk_id']}")
    print(f"Source File: {first_page['source_file']}")
    print(f"Page {first_page['page_number']}:")
    print(f"Text length: {len(first_page['text'])} characters")
    print(f"Tables found: {len(first_page['tables'])}")
    print(f"Words extracted: {len(first_page['words'])}")