import re
from typing import List, Dict, Any
from src.core.abstractions.chunking_strategy import LogicalChunkingStrategy
from src.utils.logger import get_logger
import spacy

logger = get_logger(__name__)
nlp = spacy.load("de_core_news_sm")


class HeaderBasedChunker(LogicalChunkingStrategy):
    """
    Chunk documents based on headers.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.max_chunk_size = config.get('max_chunk_size', 1500)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.min_chunk_size = config.get('min_chunk_size', 100)

    @property
    def strategy_name(self) -> str:
        return "header_based"

    def get_config_requirements(self) -> List[str]:
        return ['max_chunk_size', 'chunk_overlap', 'min_chunk_size']

    def chunk(self, structured_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not structured_data:
            return []

        chunks = []
        current_chunk_text = ""
        current_pages = []
        chunk_index = 0

        for element in structured_data:
            text = element.get('text', '').strip()
            page = element.get('page_number', 1)
            if not text:
                continue

            # Start new chunk if max size exceeded
            if len(current_chunk_text) + len(text) > self.max_chunk_size and current_chunk_text:
                chunks.append(self._create_chunk(current_chunk_text, current_pages, chunk_index, element))
                chunk_index += 1
                current_chunk_text = current_chunk_text[-self.chunk_overlap:] if self.chunk_overlap else ""
                current_pages = []

            current_chunk_text += (" " if current_chunk_text else "") + text
            current_pages.append(page)

        if current_chunk_text:
            chunks.append(self._create_chunk(current_chunk_text, current_pages, chunk_index, structured_data[-1]))

        return chunks

    def _create_chunk(self, text: str, pages: List[int], chunk_index: int, element: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'content': text.strip(),
            'metadata': {
                'page_number': min(pages) if pages else 1,
                'page_range': (min(pages), max(pages)) if pages else (1, 1),
                'section_title': element.get('header', 'Header Section'),
                'chunk_index': chunk_index,
                'contains_table': False,
                'contains_image': False,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        }


class SemanticChunker(LogicalChunkingStrategy):
    """
    Chunk documents based on semantic coherence.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.max_chunk_size = config.get('max_chunk_size', 1200)
        self.chunk_overlap = config.get('chunk_overlap', 150)
        self.min_chunk_size = config.get('min_chunk_size', 200)
        self.sentence_min_length = config.get('sentence_min_length', 10)

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def get_config_requirements(self) -> List[str]:
        return ['max_chunk_size', 'chunk_overlap', 'min_chunk_size', 'sentence_min_length']

    def chunk(self, structured_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not structured_data:
            return []

        sentences = []
        for element in structured_data:
            text = element.get('text', '').strip()
            page = element.get('page_number', 1)
            if not text:
                continue
            # German sentence splitting
            doc = nlp(text)
            for sent in doc.sents:
                sentence = sent.text.strip()
                if len(sentence) >= self.sentence_min_length:
                    sentences.append({'text': sentence, 'page': page})

        chunks = []
        current_text = ""
        current_pages = []
        chunk_index = 0

        for s_index, sentence in enumerate(sentences):
            text = sentence['text']
            page = sentence['page']
            if len(current_text) + len(text) > self.max_chunk_size and current_text:
                chunks.append(self._create_chunk(current_text, current_pages, chunk_index))
                chunk_index += 1
                current_text = current_text[-self.chunk_overlap:] if self.chunk_overlap else ""
                current_pages = []

            current_text += (" " if current_text else "") + text
            current_pages.append(page)

        if current_text:
            chunks.append(self._create_chunk(current_text, current_pages, chunk_index))

        # semantic_index
        for idx, chunk in enumerate(chunks):
            chunk['metadata']['semantic_index'] = idx

        return chunks

    def _create_chunk(self, text: str, pages: List[int], chunk_index: int) -> Dict[str, Any]:
        return {
            'content': text.strip(),
            'metadata': {
                'page_number': min(pages) if pages else 1,
                'page_range': (min(pages), max(pages)) if pages else (1, 1),
                'section_title': 'Semantic Section',
                'chunk_index': chunk_index,
                'contains_table': False,
                'contains_image': False,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        }


class ChainedChunker(LogicalChunkingStrategy):
    """
    Chains HeaderBasedChunker followed by SemanticChunker.
    This is the default chunking pipeline.
    """
    def __init__(self, header_config: Dict[str, Any] = None, semantic_config: Dict[str, Any] = None):
        self.header_chunker = HeaderBasedChunker(header_config or {
            'max_chunk_size': 1500,
            'chunk_overlap': 200,
            'min_chunk_size': 100,
            'header_font_threshold': 14.0
        })
        self.semantic_chunker = SemanticChunker(semantic_config or {
            'max_chunk_size': 1200,
            'chunk_overlap': 150,
            'min_chunk_size': 200,
            'sentence_min_length': 10
        })

    @property
    def strategy_name(self) -> str:
        return "chained"

    def chunk(self, structured_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        header_chunks = self.header_chunker.chunk(structured_data)
        logger.info(f"HeaderBasedChunker created {len(header_chunks)}")

        final_chunks = []
        for h_index, h_chunk in enumerate(header_chunks):
            semantic_chunks = self.semantic_chunker.chunk([{
                'text': h_chunk['content'],
                'page_number': h_chunk['metadata']['page_number'],
                'font_size': 12.0,
                'element_type': 'paragraph'
            }])

            for s_chunk in semantic_chunks:
                s_chunk['metadata']['header_index'] = h_index
                final_chunks.append(s_chunk)

        logger.info(f"Chained chunking produced {len(final_chunks)} final chunks")
        return final_chunks
