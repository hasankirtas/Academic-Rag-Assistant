"""
Academic RAG Assistant - Streamlit Interface
"""

import streamlit as st
import time
import io
import tempfile
import pdfplumber
from typing import Optional, Dict, List
import logging
from pathlib import Path
import hashlib
import json
import os
from huggingface_hub import HfApi
try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None
from requests.exceptions import HTTPError
import requests

from src.services.rag_service import RAGPipeline, create_rag_pipeline
from src.services.llm_service import create_llm_service
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Constant used for the default select option label
DEFAULT_PDF_OPTION = "Select a PDF"

# Page configuration
st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translation dictionary with enhanced error messages
TRANSLATIONS = {
    "de": {
        "app_title": "Akademischer RAG-Assistent",
        "subtitle": "Stellen Sie Fragen zu Ihren PDF-Dokumenten",
        "language_select": "Sprache wählen",
        "pdf_upload": "PDF-Dokument hochladen",
        "upload_help": "Laden Sie ein akademisches PDF-Dokument hoch (max. 10MB)",
        "processing": "Verarbeite PDF",
        "processing_complete": "PDF erfolgreich verarbeitet!",
        "chat_input": "Stellen Sie Ihre Frage...",
        "no_pdf": "Bitte laden Sie zuerst ein PDF-Dokument hoch.",
        "invalid_pdf": "Ungültige PDF-Datei. Bitte überprüfen Sie das Dokument.",
        "file_too_large": "Datei zu groß. Maximale Größe: 10MB",
        "processing_error": "Fehler beim Verarbeiten der PDF-Datei.",
        "example_questions": "Beispielfragen:",
        "clear_chat": "Chat löschen",
        "pipeline_info": "Pipeline-Informationen",
        "contexts_found": "Gefundene Kontexte:",
        "generation_time": "Generierungszeit:",
        "no_contexts": "Keine relevanten Kontexte gefunden.",
        "system_status": "Systemstatus",
        "ready": "Bereit",
        "processing_pdf": "PDF wird verarbeitet...",
        "embedding_failed": "Embedding-Erstellung fehlgeschlagen",
        "vector_db_error": "Vektordatenbank-Fehler",
        "file_corrupted": "PDF-Datei ist beschädigt",
        "memory_error": "Nicht genügend Speicher für diese Datei"
    },
    "tr": {
        "app_title": "Akademik RAG Asistanı",
        "subtitle": "PDF belgeleriniz hakkında sorular sorun",
        "language_select": "Dil seçin",
        "pdf_upload": "PDF belgesi yükleyin",
        "upload_help": "Akademik bir PDF belgesi yükleyin (maks. 10MB)",
        "processing": "PDF işleniyor...",
        "processing_complete": "PDF başarıyla işlendi!",
        "chat_input": "Sorunuzu yazın...",
        "no_pdf": "Lütfen önce bir PDF belgesi yükleyin.",
        "invalid_pdf": "Geçersiz PDF dosyası. Lütfen belgeyi kontrol edin.",
        "file_too_large": "Dosya çok büyük. Maksimum boyut: 10MB",
        "processing_error": "PDF dosyası işlenirken hata oluştu.",
        "example_questions": "Örnek sorular:",
        "clear_chat": "Sohbeti temizle",
        "pipeline_info": "Pipeline bilgileri",
        "contexts_found": "Bulunan bağlamlar:",
        "generation_time": "Üretim süresi:",
        "no_contexts": "İlgili bağlam bulunamadı.",
        "system_status": "Sistem durumu",
        "ready": "Hazır",
        "processing_pdf": "PDF işleniyor...",
        "embedding_failed": "Embedding oluşturma başarısız",
        "vector_db_error": "Vektör veritabanı hatası",
        "file_corrupted": "PDF dosyası bozuk",
        "memory_error": "Bu dosya için yeterli bellek yok"
    }
}

def validate_hf_token(token: str) -> bool:
    """
    Check if the Hugging Face API token is valid by making a simple API request.
    
    Args:
        token: Hugging Face API token
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    try:
        # Method 1: Try using HfApi
        api = HfApi()
        # Use whoami endpoint to validate token
        result = api.whoami(token=token)
        return result is not None
    except Exception as e:
        try:
            # Method 2: Direct API call as fallback
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                "https://huggingface.co/api/whoami",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception as fallback_error:
            logger.warning(f"Token validation failed: {str(e)}, Fallback error: {str(fallback_error)}")
            return False

def on_language_change():
    """
    Callback when user changes language.
    Ensures PDF and pipeline session state are preserved.
    """
    selected_lang_key = st.session_state.get("language_select_key", "de")
    st.session_state.language = selected_lang_key
    st.info(f"🌐 Language switched to {selected_lang_key}. PDF and pipeline state preserved.")

def get_text(key: str) -> str:
    """Get translated text based on selected language."""
    lang = st.session_state.get('language', 'de')
    return TRANSLATIONS.get(lang, TRANSLATIONS['de']).get(key, key)

def get_pdf_hash(file_content: bytes) -> str:
    """Generate a hash for PDF file content to detect duplicates."""
    return hashlib.md5(file_content).hexdigest()

def get_processed_pdfs_file() -> str:
    """Get the path to the processed PDFs tracking file."""
    return os.path.join("data", "processed_pdfs.json")

def load_processed_pdfs() -> Dict[str, Dict]:
    """Load the list of processed PDFs from file."""
    file_path = get_processed_pdfs_file()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading processed PDFs: {e}")
    return {}

def save_processed_pdf(pdf_name: str, pdf_hash: str, metadata: Dict):
    """Save processed PDF information to file."""
    file_path = get_processed_pdfs_file()
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    processed_pdfs = load_processed_pdfs()
    processed_pdfs[pdf_hash] = {
        "name": pdf_name,
        "hash": pdf_hash,
        "processed_at": time.time(),
        "metadata": metadata
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_pdfs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving processed PDF: {e}")

def is_pdf_processed(file_content: bytes) -> Optional[Dict]:
    """Check if a PDF has already been processed."""
    pdf_hash = get_pdf_hash(file_content)
    processed_pdfs = load_processed_pdfs()
    return processed_pdfs.get(pdf_hash)

# Helper functions removed - using RAG pipeline's PDF processing instead

@st.cache_resource(show_spinner=False)
def get_rag_pipeline(hf_token: str = None):
    """Cached RAG pipeline initialization."""
    try:
        # Update config with HF token if provided
        if hf_token:
            import src.utils.config_parser as config_module
            config_module.CONFIG["llm"]["api_token"] = hf_token
        return create_rag_pipeline()
    except Exception as e:
        logger.error(f"Failed to create RAG pipeline: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def process_pdf_cached(file_content: bytes, file_name: str) -> Dict:
    """
    Cached PDF processing to avoid reprocessing same files.
    
    Args:
        file_content: PDF file content as bytes
        file_name: Name of the PDF file
        
    Returns:
        Dict: Processing result with chunks and metadata
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        # Use RAG pipeline's PDF loader (pdfplumber-based)
        from src.ingestion.pdf_loader import PDFLoader
        pdf_loader = PDFLoader()
        structured_elements = pdf_loader.load(tmp_path)
        
        # Convert to simple chunks format for compatibility
        document_chunks = []
        for element in structured_elements:
            document_chunks.append({
                'text': element['text'],
                'page': element['page_number'],
                'chunk_index': len(document_chunks),
                'font_size': element.get('font_size', 12.0),
                'is_bold': element.get('is_bold', False),
                'element_type': element.get('element_type', 'paragraph')
            })
        
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)
        
        return {
            'success': True,
            'chunks': document_chunks,
            'total_pages': max([chunk['page'] for chunk in document_chunks]) if document_chunks else 0,
            'total_chunks': len(document_chunks)
        }
        
    except Exception as e:
        logger.error(f"Cached PDF processing error: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'chunks': []
        }

def create_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Create overlapping text chunks for better context preservation.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundaries
        if end < len(text):
            last_period = chunk.rfind('.')
            last_exclamation = chunk.rfind('!')
            last_question = chunk.rfind('?')
            
            sentence_end = max(last_period, last_exclamation, last_question)
            if sentence_end > chunk_size * 0.7:  # Only if we're not cutting too much
                chunk = text[start:start + sentence_end + 1]
                end = start + sentence_end + 1
        
        chunks.append(chunk.strip())
        start = end - overlap if end < len(text) else end
    
    return [chunk for chunk in chunks if chunk.strip()]

def validate_pdf(uploaded_file) -> tuple[bool, str]:
    """
    Enhanced PDF validation with specific error messages using pdfplumber.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check file size (10MB limit)
        if uploaded_file.size > 10 * 1024 * 1024:
            return False, get_text("file_too_large")
        
        # Reset file pointer
        uploaded_file.seek(0)
            
        # Check if it's a valid PDF using pdfplumber
        try:
            # Create temporary file for validation
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Try to open with pdfplumber and extract some text
            with pdfplumber.open(tmp_path) as pdf:
                # Check if we can access pages
                if len(pdf.pages) == 0:
                    return False, get_text("invalid_pdf")
                
                # Extract text from first few pages for content validation
                first_page_text = ""
                for page_num in range(min(3, len(pdf.pages))):
                    try:
                        page_text = pdf.pages[page_num].extract_text()
                        if page_text:
                            first_page_text += page_text
                    except Exception:
                        continue
            
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
            
        except Exception as e:
            if "corrupted" in str(e).lower() or "invalid" in str(e).lower():
                return False, get_text("file_corrupted")
            elif "memory" in str(e).lower():
                return False, get_text("memory_error")
            else:
                return False, get_text("invalid_pdf")
            
        # Basic German academic content check
        if first_page_text.strip():
            german_indicators = ['der', 'die', 'das', 'und', 'ist', 'werden', 'haben']
            academic_indicators = ['Theorie', 'Analyse', 'Forschung', 'Studie', 'Konzept', 'Definition']
            
            text_lower = first_page_text.lower()
            german_count = sum(1 for word in german_indicators if word in text_lower)
            academic_count = sum(1 for word in academic_indicators if word.lower() in text_lower)
            
            # Basic validation: should have some German words
            if german_count >= 2 or academic_count >= 1:
                return True, ""
            else:
                return False, get_text("invalid_pdf")
        else:
            return False, get_text("invalid_pdf")
        
    except Exception as e:
        logger.error(f"PDF validation error: {str(e)}")
        return False, get_text("processing_error")

def initialize_rag_pipeline() -> Optional[RAGPipeline]:
    """Initialize RAG pipeline with caching and error handling, with HF token validation."""
    try:
        hf_token = st.session_state.get('hf_token', '')

        # If no token is provided, warn the user
        if not hf_token:
            st.warning("⚠️ Please enter your Hugging Face API token in the sidebar")
            return None

        # Validate the token if not already validated
        if not st.session_state.get('hf_token_validated', False):
            if validate_hf_token(hf_token):
                st.session_state.hf_token_validated = True
                st.sidebar.success("✅ API Token validated")
            else:
                st.session_state.hf_token_validated = False
                st.sidebar.error("❌ Invalid Hugging Face API Token")
                return None  # Do not initialize pipeline if token is invalid

        # Check if pipeline needs to be recreated (token changed)
        current_token = st.session_state.get('current_hf_token', '')
        if 'rag_pipeline' not in st.session_state or current_token != hf_token:
            with st.spinner("Initializing RAG pipeline..."):
                st.session_state.rag_pipeline = get_rag_pipeline(hf_token)
                st.session_state.current_hf_token = hf_token

        # Ensure LLM service exists (and refresh on token change)
        if 'llm_service' not in st.session_state or current_token != hf_token:
            try:
                st.session_state.llm_service = create_llm_service()
            except Exception as e:
                st.error(f"LLM init error: {e}")
                logger.error(f"LLM init error: {e}")

        return st.session_state.rag_pipeline

    except Exception as e:
        st.error(f"Pipeline initialization error: {str(e)}")
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        return None

def _is_model_installed(model_name: str) -> bool:
    """Check if a Hugging Face model is available locally."""
    try:
        if AutoTokenizer is None or AutoModel is None:
            return False
        _ = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        _ = AutoModel.from_pretrained(model_name, local_files_only=True)
        return True
    except Exception:
        return False

def render_model_loader_sidebar():
    """Render multilingual model loader with install button and progress."""
    with st.sidebar.expander("🧩 Multilingual Model", expanded=False):
        model_name = "intfloat/multilingual-e5-base"
        if 'model_installed' not in st.session_state:
            st.session_state.model_installed = _is_model_installed(model_name)

        st.write(f"Model: `{model_name}`")

        if st.session_state.model_installed:
            st.success("✅ Model installed")
            st.button("Install Model", disabled=True)
            return

        install_clicked = st.button("⬇️ Install Model", disabled=False)
        if install_clicked:
            progress = st.progress(0)
            status = st.empty()
            try:
                # Coarse-grained progress; actual download progress is handled by transformers
                status.text("Preparing download...")
                progress.progress(10)
                if AutoTokenizer is None or AutoModel is None:
                    raise RuntimeError("Transformers not available")
                # Download tokenizer and model
                _ = AutoTokenizer.from_pretrained(model_name)
                progress.progress(70)
                status.text("Downloading model weights...")
                _ = AutoModel.from_pretrained(model_name)
                progress.progress(100)
                st.session_state.model_installed = True
                status.text("")
                st.success("✅ Model installed successfully")
            except Exception as e:
                status.text("")
                st.error(f"Model install failed: {e}")

def process_uploaded_pdf(uploaded_file, rag_pipeline: RAGPipeline) -> bool:
    """
    Enhanced PDF processing with streaming and better error handling.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        rag_pipeline: RAG pipeline instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get file content
        file_content = uploaded_file.getvalue()
        
        # Check if PDF has already been processed
        existing_pdf = is_pdf_processed(file_content)
        if existing_pdf:
            st.info(f"📋 **{uploaded_file.name}** zaten işlenmiş! ({existing_pdf['name']})")
            st.success("✅ Bu PDF daha önce işlenmiş ve veritabanında mevcut.")
            
            # Set session state to indicate PDF is ready
            st.session_state.pdf_processed = True
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.total_chunks = existing_pdf['metadata'].get('total_chunks', 0)
            
            return True
        
        # Use cached processing
        processing_result = process_pdf_cached(file_content, uploaded_file.name)
        
        if not processing_result['success']:
            error_key = "processing_error"
            if "memory" in processing_result.get('error', '').lower():
                error_key = "memory_error"
            elif "embedding" in processing_result.get('error', '').lower():
                error_key = "embedding_failed"
            elif "vector" in processing_result.get('error', '').lower():
                error_key = "vector_db_error"
            
            st.error(get_text(error_key))
            return False
        
        document_chunks = processing_result['chunks']

        # Dynamic protected term derivation removed (deprecated strategy)
        
        # Real PDF processing with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Clean text and generate embeddings with error handling
        status_text.text("Cleaning text and generating embeddings...")
        progress_bar.progress(30)
        
        embeddings = []
        failed_chunks = 0
        cleaned_chunks = []

        # Initialize cleaner
        try:
            from src.preprocessing.text_cleaner import TextCleaner
            cleaner = TextCleaner()
        except Exception as e:
            cleaner = None
            logger.warning(f"TextCleaner init failed: {e}")
        
        for i, chunk in enumerate(document_chunks):
            try:
                text_to_embed = chunk['text']
                if cleaner:
                    cleaned = cleaner.clean(text_to_embed)
                    text_to_embed = cleaned.get('text', text_to_embed)
                # Skip only truly empty texts to avoid invalid embeddings
                if not isinstance(text_to_embed, str) or not text_to_embed.strip():
                    logger.debug(f"Skipping chunk {i} due to empty text")
                    failed_chunks += 1
                    embeddings.append(None)
                    cleaned_chunks.append(chunk)
                    continue
                embedding = rag_pipeline.embedding_service.embed_text(text_to_embed)
                # Validate embedding content
                if not isinstance(embedding, list) or len(embedding) == 0:
                    logger.warning(f"Received empty embedding for chunk {i}, skipping")
                    failed_chunks += 1
                    embeddings.append(None)
                    cleaned_chunks.append(chunk)
                    continue
                embeddings.append(embedding)
                cleaned_chunks.append({**chunk, 'text': text_to_embed})
                
                # Update progress
                progress = 30 + (i / len(document_chunks)) * 40
                progress_bar.progress(int(progress))
                
            except Exception as e:
                logger.warning(f"Failed to embed chunk {i}: {str(e)}")
                failed_chunks += 1
                embeddings.append(None)  # Placeholder
                cleaned_chunks.append(chunk)
        
        # Filter out failed/empty embeddings
        valid_chunks = []
        valid_embeddings = []
        for chunk, embedding in zip(cleaned_chunks, embeddings):
            if isinstance(embedding, list) and len(embedding) > 0:
                valid_chunks.append(chunk)
                valid_embeddings.append(embedding)
        
        if len(valid_chunks) == 0:
            st.error(get_text("embedding_failed"))
            return False
        
        progress_bar.progress(70)
        
        # Step 2: Store in vector database with error handling and consistency checks
        status_text.text("Storing in vector database...")
        
        try:
            # Convert to DocumentChunk format
            from src.implementations.vector_db.vector_database_service import DocumentChunk
            formatted_chunks = [
                DocumentChunk(
                    text=chunk['text'],
                    page=chunk['page'],
                    chunk_index=chunk['chunk_index']
                ) for chunk in valid_chunks
            ]
            # Ensure lengths match
            if len(formatted_chunks) != len(valid_embeddings):
                logger.error(f"Mismatch between chunks ({len(formatted_chunks)}) and embeddings ({len(valid_embeddings)})")
                st.error(get_text("vector_db_error"))
                return False
            
            # Store in vector database
            success = rag_pipeline.vector_db_service.store_document_chunks(
                chunks=formatted_chunks,
                embeddings=valid_embeddings
            )
            
            if not success:
                st.error(get_text("vector_db_error"))
                return False
                
        except Exception as e:
            logger.error(f"Vector DB storage error: {str(e)}")
            st.error(get_text("vector_db_error"))
            return False
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        # Show processing statistics
        if failed_chunks > 0:
            st.warning(f"⚠️ {failed_chunks} chunks failed to process, but {len(valid_chunks)} were successful.")
        
        progress_bar.empty()
        status_text.empty()
        
        # Save processed PDF metadata
        pdf_hash = get_pdf_hash(file_content)
        metadata = {
            'total_chunks': len(valid_chunks),
            'total_pages': processing_result.get('total_pages', 0),
            'file_size': len(file_content),
            'failed_chunks': failed_chunks
        }
        save_processed_pdf(uploaded_file.name, pdf_hash, metadata)
        
        return True
        
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        st.error(f"Processing error: {str(e)}")
        return False

def display_chat_interface():
    """Display the main chat interface with enhanced features."""
    
    # Chat container
    chat_container = st.container()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display contexts if available
                if message["role"] == "assistant" and "contexts" in message:
                    if message["contexts"]:
                        with st.expander(f"{get_text('contexts_found')} ({len(message['contexts'])})"):
                            for i, ctx in enumerate(message["contexts"], 1):
                                # Handle both 'content' and 'text' keys for backward compatibility
                                content = ctx.get('content', ctx.get('text', ''))
                                st.write(f"**{i}.** {content[:200]}...")
                                st.caption(f"Score: {ctx.get('hybrid_score', ctx.get('score', 0)):.3f} | Page: {ctx.get('page', 'N/A')}")
                    
                    # Show generation time and tokens
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if "generation_time" in message:
                            st.caption(f"{get_text('generation_time')} {message['generation_time']:.0f}ms")
                    with col2:
                        st.caption(f"{get_text('contexts_found')} {len(message.get('contexts', []))}")
                    with col3:
                        if message.get('llm_tokens'):
                            st.caption(f"Tokens: {message['llm_tokens']}")
    
    # Chat input
    if prompt := st.chat_input(get_text("chat_input")):
        # Check if PDF is loaded
        if 'pdf_processed' not in st.session_state or not st.session_state.pdf_processed:
            st.warning(get_text("no_pdf"))
            return
        
        # Check if LLM service is available
        if 'llm_service' not in st.session_state:
            st.error("LLM service not initialized. Please check configuration.")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response (streaming)
        with st.chat_message("assistant"):
            try:
                rag_pipeline = st.session_state.rag_pipeline
                llm_service = st.session_state.llm_service

                # Query classification to skip RAG for meta/general questions
                def should_use_rag(q: str) -> bool:
                    ql = q.strip().lower()
                    meta_patterns = [
                        "pdf'i", "pdfi", "pdfi gördün", "pdf'i gördün", "dosyayı gördün",
                        "pdf yüklü mü", "pdfi yükledim", "ben kimim", "selam", "merhaba",
                        "sen kimsin", "model", "versiyon", "token", "hata", "bug",
                        "did you see the pdf", "see the pdf", "are you loaded",
                    ]
                    return not any(p in ql for p in meta_patterns)

                use_rag = should_use_rag(prompt)
                start_time = time.time()
                contexts = []
                if use_rag:
                    rag_result = rag_pipeline.query(prompt, k=10)
                    contexts = rag_result.get('contexts', []) if rag_result.get('success', False) else []

                # Adapt contexts to LLM format
                llm_contexts = [
                    {
                        'text': ctx.get('content', ctx.get('text', '')),
                        'hybrid_score': ctx.get('hybrid_score', ctx.get('score', 0))
                    } for ctx in contexts
                ]

                # Streaming if available; fallback to non-streaming for older cached sessions
                simulate_stream = st.session_state.get('simulate_stream', True)

                if hasattr(llm_service, "stream_response") and not simulate_stream:
                    if hasattr(st, "write_stream"):
                        # Use Streamlit's native streaming renderer with character-level splitting
                        def _char_stream():
                            for chunk in llm_service.stream_response(query=prompt, contexts=llm_contexts):
                                if not chunk:
                                    continue
                                for ch in chunk:
                                    yield ch
                        final_answer = st.write_stream(_char_stream())
                    else:
                        # Fallback manual streaming
                        placeholder = st.empty()
                        streamed_text = ""
                        for chunk in llm_service.stream_response(query=prompt, contexts=llm_contexts):
                            if not chunk:
                                continue
                            for ch in chunk:
                                streamed_text += ch
                                placeholder.markdown(streamed_text)
                        final_answer = streamed_text
                else:
                    # Non-streaming backend or simulate client-side typing
                    with st.spinner("Generating response..."):
                        result = llm_service.generate_response(query=prompt, contexts=llm_contexts)
                    full_text = result.get("answer", "")
                    if simulate_stream:
                        placeholder = st.empty()
                        rendered = ""
                        for ch in full_text:
                            rendered += ch
                            placeholder.markdown(rendered)
                            time.sleep(0.002)
                        final_answer = rendered
                    else:
                        st.markdown(full_text)
                        final_answer = full_text

                generation_time = (time.time() - start_time) * 1000

                # Persist in history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_answer,
                    "contexts": contexts,
                    "generation_time": generation_time,
                })

                # Display generation info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"{get_text('generation_time')} {generation_time:.0f}ms")
                with col2:
                    st.caption(f"{get_text('contexts_found')} {len(contexts)}")
                with col3:
                    pass

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.error(f"Chat error: {str(e)}")
                
                # Add error to chat history
                error_response = f"Error: {str(e)}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_response,
                    "contexts": [],
                    "generation_time": 0,
                    "error": str(e)
                })

def render_about_panel():

    """Render an informational 'About the System' panel (English)."""
    st.header("ℹ️ About the System")

    tabs = st.tabs([
        "General",
        "Architecture",
        "Models & Settings",
        "Data & Metrics",
        "FAQ",
        "Changelog",
    ])

    with tabs[0]:
        st.subheader("Purpose & Scope")
        st.write(
            "A RAG-based assistant for German academic PDFs: fast search, citations, and explanatory answers."
        )
        st.subheader("Highlights")
        st.markdown(
            "- Retrieval-Augmented Generation (RAG)\n"
            "- German text cleaning & logical chunking\n"
            "- Vector index via Chroma\n"
            "- Context-aware answers with source hints"
        )
        with st.expander("Transparency & Privacy"):
            st.markdown(
                "- Local processing; vector index stored in Chroma\n"
                "- No sharing of personal data\n"
                "- Remove/rebuild the index from within the app (if enabled)"
            )

    with tabs[1]:
        st.subheader("Data Flow (High-Level)")
        st.code(
            "PDF → Cleaning → Chunking → Embedding → Chroma (Vector DB) → Retrieval → LLM",
            language="text",
        )
        st.caption("Overview of the main pipeline stages")

    with tabs[2]:
        st.subheader("Configuration Overview")
        try:
            rag_pipeline = st.session_state.get("rag_pipeline")
            if rag_pipeline is not None:
                info = rag_pipeline.get_pipeline_info()
                st.json(info)
            else:
                st.info("Pipeline not initialized yet.")
        except Exception as e:
            st.error(f"Failed to fetch configuration: {e}")

    with tabs[3]:
        st.subheader("Live Metrics")
        col1, col2, col3 = st.columns(3)
        total_chunks = st.session_state.get("total_chunks", 0)
        processed_flag = st.session_state.get("pdf_processed", False)
        processed_name = st.session_state.get("current_pdf", "—")
        processed_pdfs = load_processed_pdfs()
        with col1:
            st.metric("Processed Chunks", value=total_chunks)
        with col2:
            st.metric("PDF Loaded", value="Yes" if processed_flag else "No")
        with col3:
            st.metric("Stored PDFs", value=len(processed_pdfs))
        st.caption(f"Active PDF: {processed_name}")

    with tabs[4]:
        st.subheader("Frequently Asked Questions")
        with st.expander("How do I get better answers?"):
            st.markdown(
                "- Ask precise questions with domain terms\n"
                "- Mention relevant chapters/pages\n"
                "- Try alternative phrasings when needed"
            )
        with st.expander("What PDFs work best?"):
            st.markdown(
                "- Text-based German academic PDFs (not image-only scans)\n"
                "- Clear structure (headings), clean layout"
            )

    with tabs[5]:
        st.subheader("Changelog")
        st.markdown(
            "- v0.2.0: Added metrics & About-the-System panel\n"
            "- v0.1.0: Initial release with RAG pipeline"
        )

def main():
    """Main Streamlit application."""
    
    # Hugging Face API Key Input
    st.sidebar.header("🔑 API Configuration")
    hf_token = st.sidebar.text_input(
        "Hugging Face API Token",
        type="password",
        help="Enter your Hugging Face API token to use the LLM service",
        value=st.session_state.get('hf_token', '')
    )
    
    if hf_token:
        st.session_state.hf_token = hf_token
        st.sidebar.success("✅ API Token configured")
    else:
        st.sidebar.warning("⚠️ Please enter your Hugging Face API token")
        st.sidebar.markdown("""
        **How to get your API token:**
        1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
        2. Create a new token
        3. Copy and paste it here
        """)
    
    # Enhanced CSS with dark mode and responsive design
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .stButton > button {
        background-color: #1f4e79;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #163d63;
        transform: translateY(-1px);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .css-1d391kg {
            background-color: #2d2d2d;
        }
        .success-box {
            background-color: #1a472a;
            border-color: #28a745;
            color: #d4edda;
        }
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stButton > button {
            font-size: 14px;
            padding: 0.3rem 0.8rem;
        }
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        # Force German language (Chat runs only in German!)
        st.session_state.language = 'de'
        
        st.divider()

        # Multilingual model loader controls
        render_model_loader_sidebar()

        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            for key in [
                'hf_token', 'hf_token_validated', 'current_hf_token', 'rag_pipeline',
                'llm_service', 'pdf_processed', 'current_pdf', 'total_chunks',
                'chat_history', 'example_questions', 'processed_pdf_select'
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Logged out. Session cleared.")
            st.rerun()
        
        # System status
        st.header(f"📊 {get_text('system_status')}")
        
        # Initialize RAG pipeline
        rag_pipeline = initialize_rag_pipeline()
        
        if rag_pipeline:
            st.success(f"✅ {get_text('ready')}")
            
            # Display pipeline info
            with st.expander(f"🔧 {get_text('pipeline_info')}"):
                try:
                    info = rag_pipeline.get_pipeline_info()
                    for key, value in info.items():
                        if isinstance(value, dict):
                            st.write(f"**{key}:**")
                            for sub_key, sub_value in value.items():
                                st.write(f"  - {sub_key}: {sub_value}")
                        else:
                            st.write(f"**{key}:** {value}")
                except Exception as e:
                    st.error(f"Error getting pipeline info: {str(e)}")
        else:
            st.error("❌ Pipeline initialization failed")
            return
        
        # PDF processing status
        if st.session_state.get('pdf_processed', False):
            st.info(f"📖 **PDF:** {st.session_state.get('current_pdf', 'Unknown')}")
            st.info(f"📊 **Chunks:** {st.session_state.get('total_chunks', 'Unknown')}")
        
        st.divider()
        
        # Processed PDFs list
        st.header("📚 Processed PDFs")
        processed_pdfs = load_processed_pdfs()
        
        # Ensure selectbox state can be reset safely before widget instantiation
        if st.session_state.get("reset_processed_pdf_select", False):
            st.session_state.pop("processed_pdf_select", None)
            st.session_state.reset_processed_pdf_select = False
        
        if processed_pdfs:
            pdf_names = [pdf_info['name'] for pdf_info in processed_pdfs.values()]
            selected_pdf = st.selectbox(
                "Select a PDF to chat with:",
                options=[DEFAULT_PDF_OPTION] + pdf_names,
                index=0,
                key="processed_pdf_select",
            )
            
            # Only act when a real PDF (not the default option) is selected
            if selected_pdf and selected_pdf != DEFAULT_PDF_OPTION:
                # Find the selected PDF info
                selected_pdf_info = None
                for pdf_info in processed_pdfs.values():
                    if pdf_info['name'] == selected_pdf:
                        selected_pdf_info = pdf_info
                        break
                
                if selected_pdf_info:
                    st.success(f"✅ Selected: {selected_pdf}")
                    st.info(f"📊 Chunks: {selected_pdf_info['metadata'].get('total_chunks', 0)}")
                    st.info(f"📄 Pages: {selected_pdf_info['metadata'].get('total_pages', 0)}")
                    
                    # Set session state for selected PDF
                    st.session_state.pdf_processed = True
                    st.session_state.current_pdf = selected_pdf
                    st.session_state.total_chunks = selected_pdf_info['metadata'].get('total_chunks', 0)
                    
                    if st.button("🗑️ Remove from list", type="secondary"):
                        # Remove PDF from list
                        pdf_hash = selected_pdf_info['hash']
                        del processed_pdfs[pdf_hash]
                        
                        # Save updated list
                        file_path = get_processed_pdfs_file()
                        try:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(processed_pdfs, f, indent=2, ensure_ascii=False)
                            st.success("PDF removed from list!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing PDF: {e}")
        else:
            st.info("No PDFs processed yet")
        
        st.divider()
        
        # Streaming is always enabled by default; no user toggle
        if 'simulate_stream' not in st.session_state:
            # Prefer server-side streaming when available
            st.session_state.simulate_stream = False
        
        # Clear chat button
        if st.button(f"🗑️ {get_text('clear_chat')}", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # About panel in sidebar as collapsible section
        with st.expander("ℹ️ About the System", expanded=False):
            render_about_panel()
    
    # Main content
    st.title(f"📚 {get_text('app_title')}")
    st.markdown(f"*{get_text('subtitle')}*")
    
    # PDF Upload Section (always keep upload active)
    if not st.session_state.get('pdf_processed', False):
        st.header(f"📄 {get_text('pdf_upload')}")
    else:
        st.header("📄 PDF Status")
        st.success(f"✅ **Active PDF:** {st.session_state.get('current_pdf', 'Unknown')}")
        st.info(f"📊 **Chunks:** {st.session_state.get('total_chunks', 0)}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Upload New PDF", use_container_width=True):
                # Reset current PDF-related state and sidebar selection
                st.session_state.pdf_processed = False
                st.session_state.current_pdf = None
                # Defer widget reset to the next run before selectbox creation
                st.session_state.reset_processed_pdf_select = True
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh PDF List", use_container_width=True):
                st.rerun()
        
    # File uploader remains available regardless of current selection
    uploaded_file = st.file_uploader(
        get_text("upload_help"),
        type=['pdf'],
        help=get_text("upload_help")
    )
    
    if uploaded_file is not None:
        # File validation with enhanced error handling
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            is_valid, error_msg = validate_pdf(uploaded_file)
            
            if is_valid:
                st.info(f"📋 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
                
                # Process PDF button
                if st.button("🚀 " + get_text("processing"), use_container_width=True):
                    success = process_uploaded_pdf(uploaded_file, rag_pipeline)
                    
                    if success:
                        st.session_state.pdf_processed = True
                        st.session_state.current_pdf = uploaded_file.name
                        
                        # Get processing stats from cache
                        file_content = uploaded_file.getvalue()
                        cached_result = process_pdf_cached(file_content, uploaded_file.name)
                        st.session_state.total_chunks = cached_result.get('total_chunks', 0)
                        
                        st.markdown(f'<div class="success-box">✅ {get_text("processing_complete")}</div>', 
                                  unsafe_allow_html=True)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(get_text("processing_error"))
            else:
                st.error(error_msg)
    
    st.divider()
    
    # Example questions
    if st.session_state.get('pdf_processed', False):
        st.header(f"💡 {get_text('example_questions')}")
        
        # Generate example questions from current PDF content using LLM (once per session)
        if 'example_questions' not in st.session_state and 'llm_service' in st.session_state:
            try:
                llm_service = st.session_state.llm_service
                # Sample a few chunks from vector DB or cached processed file
                sample_contexts = []
                try:
                    # Attempt to retrieve generic contexts for overview
                    rag_pipeline = st.session_state.rag_pipeline
                    generic = rag_pipeline.query("Dokument Überblick und Hauptthemen", k=10)
                    sample_contexts = [
                        {
                            'text': ctx.get('content', ctx.get('text', '')),
                            'hybrid_score': ctx.get('hybrid_score', ctx.get('score', 0))
                        } for ctx in generic.get('contexts', [])
                    ]
                except Exception:
                    sample_contexts = []

                prompt = (
                    "Erzeuge 4 prägnante, inhaltsspezifische Beispielfragen zum hochgeladenen "
                    "deutschen akademischen PDF. Fokussiere auf Kernbegriffe, Definitionen, Modelle "
                    "und Zusammenhänge. Antworte als Liste, eine Frage pro Zeile, ohne Nummerierung."
                )
                gen = llm_service.generate_response(query=prompt, contexts=sample_contexts)
                lines = [x.strip("- •\t ") for x in gen.get('answer', '').splitlines() if x.strip()]
                # Keep 4 non-empty lines
                st.session_state.example_questions = lines[:4] if len(lines) >= 4 else lines
            except Exception as e:
                logger.warning(f"Example question generation failed: {e}")
                st.session_state.example_questions = []

        example_questions = [
            "Was ist das Bruttoinlandsprodukt und wie wird es berechnet?",
            "Erklären Sie die wichtigsten Prinzipien des Managements.",
            "Was sind die Unterschiede zwischen Mikro- und Makroökonomie?",
            "Wie funktioniert das Konzept der Inflation?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            col = cols[i % 2]
            with col:
                if st.button(question, key=f"example_{i}", use_container_width=True):
                    st.session_state.example_question = question
                    st.rerun()
        
        # Handle example question selection (run RAG + LLM)
        if 'example_question' in st.session_state:
            prompt = st.session_state.example_question
            del st.session_state.example_question
            
            # Add to chat history and process
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Generate response using RAG for retrieval and LLM for answer
            try:
                start_time = time.time()
                rag_result = rag_pipeline.query(prompt, k=10)
                contexts = rag_result.get('contexts', []) if rag_result.get('success', False) else []
                llm_contexts = [
                    {
                        'text': ctx.get('content', ctx.get('text', '')),
                        'hybrid_score': ctx.get('hybrid_score', ctx.get('score', 0))
                    } for ctx in contexts
                ]

                llm_service = st.session_state.get('llm_service')
                answer_text = ""
                if llm_service is not None:
                    # Use non-streaming generation here; chat view will render history
                    llm_result = llm_service.generate_response(query=prompt, contexts=llm_contexts)
                    answer_text = llm_result.get('answer', '')
                else:
                    # Fallback to any answer provided by RAG (if present)
                    answer_text = rag_result.get('answer', '')

                generation_time = (time.time() - start_time) * 1000

                # Add assistant response (with contexts)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "contexts": contexts,
                    "generation_time": generation_time
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
            
            st.rerun()
    
    st.divider()
    
    # Chat Interface
    st.header("💬 Chat")
    
    if st.session_state.get('pdf_processed', False):
        display_chat_interface()
    else:
        st.info(get_text("no_pdf"))
        
        # Show helpful tips when no PDF is loaded
        with st.expander("📖 Tips for better results", expanded=False):
            st.markdown("""
            - Upload German academic PDFs for best results
            - Ensure PDFs contain text (not just images)  
            - Questions should be specific and clear
            - Try different question formulations for better answers
            """)
    
    # Über das System now available in the sidebar expander

    # Footer with enhanced styling
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; font-size: 0.8em; padding: 1rem;'>
        <strong>Academic RAG Assistant</strong><br>
        Powered by Llama 3.1 & Multilingual E5 | Built with Streamlit<br>
        <small>Optimized for German academic documents | Developed by Hasan Kırtaş</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Initialize session state
    if 'language' not in st.session_state:
        st.session_state.language = 'de'
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    main()
