import pdfplumber
from typing import List, Dict, Optional
from pathlib import Path
from src.utils.logger import setup_logger
import uuid

logger = setup_logger(__name__)

class PDFLoader:
    """
    Extracts text and structural information (tables, heading possibilities) from PDF
    and converts it to chunker-friendly format.
    """
    
    def __init__(self):
        self.pdf_path = None

    def load(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Loads the PDF and processes each page into structured elements suitable for chunking.
        
        Returns:
            List[Dict]: Each element represents a paragraph/line with metadata:
                - 'text': str - text content
                - 'page_number': int
                - 'font_size': float
                - 'is_bold': bool
                - 'element_type': str (paragraph/header/table)
                - Additional fields: 'chunk_id', 'source_file'
        """
        self.pdf_path = pdf_path
        structured_elements = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.info(f"Processing page {page_num}...")

                    # Extract text lines
                    words = page.extract_words(extra_attrs=["size", "fontname"])
                    lines = self._group_words_to_lines(words)

                    for line in lines:
                        structured_elements.append({
                            "chunk_id": str(uuid.uuid4()),
                            "source_file": Path(pdf_path).name,
                            "page_number": page_num,
                            "text": line["text"],
                            "font_size": line.get("font_size", 12.0),
                            "is_bold": line.get("is_bold", False),
                            "element_type": line.get("element_type", "paragraph")
                        })

                    # Extract tables separately
                    tables = self._extract_tables_from_page(page)
                    for table_text in tables:
                        structured_elements.append({
                            "chunk_id": str(uuid.uuid4()),
                            "source_file": Path(pdf_path).name,
                            "page_number": page_num,
                            "text": table_text,
                            "font_size": 12.0,
                            "is_bold": False,
                            "element_type": "table"
                        })

        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            raise e

        return structured_elements

    def _group_words_to_lines(self, words: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Groups extracted words into lines and assigns font metadata.
        """
        lines_dict = {}
        for word in words:
            top = round(word['top'])
            text = word['text']
            font_size = word.get('size', 12.0)
            fontname = word.get('fontname', '')
            is_bold = 'Bold' in fontname

            if top not in lines_dict:
                lines_dict[top] = {"text": text, "font_size": font_size, "is_bold": is_bold}
            else:
                lines_dict[top]["text"] += " " + text

        # Determine element_type by font size heuristics (simple approach)
        elements = []
        for line in lines_dict.values():
            if line["font_size"] >= 14:
                element_type = "header"
            else:
                element_type = "paragraph"
            line["element_type"] = element_type
            elements.append(line)

        return elements

    def _extract_tables_from_page(self, page) -> List[str]:
        """
        Extracts tables on a page and converts them to text.
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


# Test
if __name__ == "__main__":
    loader = PDFLoader()
    documents = loader.load("data/raw/Makrooekonomie.pdf")

    first_page = documents[0]
    print(f"Chunk ID: {first_page['chunk_id']}")
    print(f"Source File: {first_page['source_file']}")
    print(f"Page {first_page['page_number']}:")
    print(f"Text length: {len(first_page['text'])} characters")
    print(f"Element type: {first_page['element_type']}")
