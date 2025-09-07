"""
Document chunking service for German academic PDF processing.
"""

from typing import List, Dict, Any, Optional
import logging

from src.core.abstractions.chunking_strategy import LogicalChunkingStrategy
from src.core.factories.chunking_factory import (
    get_chunker,
    create_chunker_from_config,
    validate_chunking_config
)
from src.utils.config_parser import load_config
from src.utils.logger import get_logger


class DocumentChunker:
    """
    Main service class for document chunking operations.
    """

    def __init__(
        self,
        strategy: Optional[LogicalChunkingStrategy] = None,
        config_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or get_logger(__name__)

        # Load configuration
        try:
            if config_path:
                self.config = load_config(config_path)
            else:
                self.config = load_config()
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ValueError(f"Configuration loading failed: {e}")

        # Initialize chunking strategy
        if strategy:
            self.strategy = strategy
            self.logger.info(f"Using provided chunking strategy: {strategy.strategy_name}")
        else:
            self.strategy = self._create_strategy_from_config()

        # Cache the performance metrics
        self._chunking_stats = {
            'total_chunks_created': 0,
            'total_documents_processed': 0,
            'average_chunk_size': 0,
            'strategy_used': self.strategy.strategy_name
        }

    def chunk_text(self, structured_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(structured_data, list):
            raise ValueError("structured_data must be a list of dictionaries.")

        if not structured_data:
            self.logger.warning("Empty structured_data provided to chunk_text.")
            return []

        self.logger.info(f"Starting chunking process with {len(structured_data)} elements.")
        self.logger.debug(f"Using strategy: {self.strategy.strategy_name}")

        try:
            # Validate input data structure
            self._validate_structured_data(structured_data)

            # Perform chunking using the configured strategy
            chunks = self.strategy.chunk(structured_data)

            # Post-process and validate chunks
            processed_chunks = self._post_process_chunks(chunks)

            # Update statistics
            self._update_chunking_stats(processed_chunks)

            self.logger.info(
                f"Chunking completed successfully. Created {len(processed_chunks)} chunks "
                f"from {len(structured_data)} elements."
            )

            return processed_chunks

        except Exception as e:
            self.logger.error(f"Chunking failed: {str(e)}")
            raise Exception(f"Document chunking failed: {str(e)}") from e

    def chunk_batch(self, documents: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        if not isinstance(documents, list):
            raise ValueError("documents must be a list of document structures.")

        if not documents:
            self.logger.warning("Empty documents list provided to chunk_batch.")
            return []

        self.logger.info(f"Starting batch chunking of {len(documents)} documents.")

        chunked_documents = []
        failed_documents = 0

        for i, document in enumerate(documents):
            try:
                chunks = self.chunk_text(document)
                chunked_documents.append(chunks)
            except Exception as e:
                self.logger.error(f"Failed to chunk document {i+1}/{len(documents)}: {e}")
                chunked_documents.append([])
                failed_documents += 1

        success_rate = ((len(documents) - failed_documents) / len(documents)) * 100
        self.logger.info(
            f"Batch chunking completed. Success rate: {success_rate:.1f}% "
            f"({len(documents) - failed_documents}/{len(documents)} documents)"
        )

        return chunked_documents

    def get_chunking_stats(self) -> Dict[str, Any]:
        return self._chunking_stats.copy()

    def update_strategy(self, new_strategy: LogicalChunkingStrategy) -> None:
        old_strategy_name = self.strategy.strategy_name
        self.strategy = new_strategy
        self._chunking_stats['strategy_used'] = new_strategy.strategy_name
        self.logger.info(f"Updated chunking strategy from {old_strategy_name} to {new_strategy.strategy_name}")

    def preview_chunking(self, structured_data: List[Dict[str, Any]], max_preview_length: int = 200) -> Dict[str, Any]:
        if not structured_data:
            return {"error": "No data provided for preview"}

        try:
            chunks = self.strategy.chunk(structured_data[:10])

            preview_chunks = []
            for i, chunk in enumerate(chunks[:3]):
                content = chunk.get('content', '')
                preview_content = content[:max_preview_length]
                if len(content) > max_preview_length:
                    preview_content += "..."

                preview_chunks.append({
                    'chunk_index': i,
                    'content_preview': preview_content,
                    'metadata_summary': {
                        'word_count': chunk.get('metadata', {}).get('word_count', 0),
                        'char_count': chunk.get('metadata', {}).get('char_count', 0),
                        'page_range': chunk.get('metadata', {}).get('page_range', (1, 1))
                    }
                })

            if chunks:
                avg_elements_per_chunk = len(structured_data[:10]) / len(chunks)
                estimated_total_chunks = max(1, int(len(structured_data) / avg_elements_per_chunk))
            else:
                estimated_total_chunks = 0

            return {
                'strategy_used': self.strategy.strategy_name,
                'total_elements': len(structured_data),
                'estimated_total_chunks': estimated_total_chunks,
                'preview_chunks': preview_chunks,
                'config_used': self._get_strategy_config_summary()
            }

        except Exception as e:
            self.logger.error(f"Preview generation failed: {e}")
            return {"error": f"Preview failed: {str(e)}"}

    def _create_strategy_from_config(self) -> LogicalChunkingStrategy:
        try:
            text_splitter_config = self.config.get('text_splitter', {})

            if not text_splitter_config:
                self.logger.info("No text_splitter configuration found, using default ChainedChunker")
                return LogicalChunkingStrategy.ChainedChunker()  # <-- default olarak ChainedChunker

            validate_chunking_config(text_splitter_config)
            strategy = create_chunker_from_config(text_splitter_config)
            self.logger.info(f"Created chunking strategy from config: {strategy.strategy_name}")
            return strategy

        except Exception as e:
            self.logger.warning(f"Failed to create strategy from config: {e}. Using default ChainedChunker.")
            return LogicalChunkingStrategy.ChainedChunker()

    def _validate_structured_data(self, structured_data: List[Dict[str, Any]]) -> None:
        if not structured_data:
            raise ValueError("structured_data is empty")

        required_fields = ['text']
        optional_fields = ['page_number', 'font_size', 'is_bold', 'element_type', 'bbox']

        for i, element in enumerate(structured_data[:5]):
            if not isinstance(element, dict):
                raise ValueError(f"Element {i} is not a dictionary")

            for field in required_fields:
                if field not in element:
                    raise ValueError(f"Element {i} missing required field '{field}'")

            text = element.get('text', '')
            if not isinstance(text, str):
                raise ValueError(f"Element {i} 'text' field must be a string")

    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                if 'content' not in chunk or 'metadata' not in chunk:
                    self.logger.warning(f"Chunk {i} missing required fields, skipping")
                    continue

                content = chunk['content'].strip()
                if len(content) < 10:
                    self.logger.debug(f"Skipping chunk {i} due to insufficient content length")
                    continue

                metadata = chunk['metadata'].copy()
                metadata.setdefault('chunk_index', i)
                metadata.setdefault('processing_timestamp', None)

                processed_chunks.append({
                    'content': content,
                    'metadata': metadata
                })

            except Exception as e:
                self.logger.error(f"Error processing chunk {i}: {e}")
                continue

        return processed_chunks

    def _update_chunking_stats(self, chunks: List[Dict[str, Any]]) -> None:
        self._chunking_stats['total_chunks_created'] += len(chunks)
        self._chunking_stats['total_documents_processed'] += 1

        if chunks:
            total_chars = sum(len(chunk['content']) for chunk in chunks)
            current_avg = total_chars / len(chunks)

            total_chunks = self._chunking_stats['total_chunks_created']
            prev_avg = self._chunking_stats['average_chunk_size']

            if total_chunks > len(chunks):
                self._chunking_stats['average_chunk_size'] = (
                    (prev_avg * (total_chunks - len(chunks)) + total_chars) / total_chunks
                )
            else:
                self._chunking_stats['average_chunk_size'] = current_avg

    def _get_strategy_config_summary(self) -> Dict[str, Any]:
        try:
            text_splitter_config = self.config.get('text_splitter', {})
            return {
                'strategy': text_splitter_config.get('strategy', 'unknown'),
                'parameters': text_splitter_config.get('parameters', {})
            }
        except Exception:
            return {'strategy': self.strategy.strategy_name, 'parameters': 'unknown'}


# Example usage and testing
if __name__ == "__main__":
    sample_structured_data = [
        {'text': 'Kapitel 1: Einführung in die Volkswirtschaftslehre', 'page_number': 1, 'font_size': 18.0, 'is_bold': True, 'element_type': 'header'},
        {'text': 'Die Volkswirtschaftslehre untersucht wirtschaftliche Prozesse.', 'page_number': 1, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},
        {'text': '1.1 Grundlegende Konzepte', 'page_number': 1, 'font_size': 16.0, 'is_bold': True, 'element_type': 'subheader'},
        {'text': 'Das Bruttoinlandsprodukt (BIP) ist ein zentraler Indikator.', 'page_number': 2, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},
        {'text': '1.2 Wirtschaftskreislauf', 'page_number': 2, 'font_size': 16.0, 'is_bold': True, 'element_type': 'subheader'},
        {'text': 'Der Wirtschaftskreislauf zeigt, wie Geld und Güter zirkulieren.', 'page_number': 2, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},

        {'text': 'Kapitel 2: Angebot und Nachfrage', 'page_number': 3, 'font_size': 18.0, 'is_bold': True, 'element_type': 'header'},
        {'text': 'Angebot und Nachfrage bestimmen Preise auf Märkten.', 'page_number': 3, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},
        {'text': '2.1 Angebotskurve', 'page_number': 3, 'font_size': 16.0, 'is_bold': True, 'element_type': 'subheader'},
        {'text': 'Die Angebotskurve zeigt die angebotene Menge bei verschiedenen Preisen.', 'page_number': 4, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},
        {'text': '2.2 Nachfragekurve', 'page_number': 4, 'font_size': 16.0, 'is_bold': True, 'element_type': 'subheader'},
        {'text': 'Die Nachfragekurve zeigt die nachgefragte Menge bei verschiedenen Preisen.', 'page_number': 4, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},

        {'text': 'Kapitel 3: Marktgleichgewicht', 'page_number': 5, 'font_size': 18.0, 'is_bold': True, 'element_type': 'header'},
        {'text': 'Das Marktgleichgewicht tritt auf, wenn Angebot und Nachfrage gleich sind.', 'page_number': 5, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},
        {'text': '3.1 Preisbildung', 'page_number': 5, 'font_size': 16.0, 'is_bold': True, 'element_type': 'subheader'},
        {'text': 'Preise passen sich an, bis der Markt im Gleichgewicht ist.', 'page_number': 6, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},
    ]
    
    print("Testing DocumentChunker with sample German academic text...")
    print("="*60)
    
    try:
        # Test with default configuration
        print("1. Testing with header-based chunking:")
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(sample_structured_data)
        
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"Content: {chunk['content'][:100]}...")
            print(f"Metadata: {chunk['metadata']}")
        
        # Test preview functionality
        print("\n" + "="*60)
        print("2. Testing preview functionality:")
        preview = chunker.preview_chunking(sample_structured_data)
        print(f"Preview result: {preview}")
        
        # Test statistics
        print("\n" + "="*60)
        print("3. Chunking statistics:")
        stats = chunker.get_chunking_stats()
        print(f"Stats: {stats}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()