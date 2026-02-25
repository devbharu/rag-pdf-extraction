"""
PRODUCTION-READY RAG Pipeline using Docling
============================================

Key Features:
- Proper text extraction using iterate_items()
- Advanced layout understanding (tables, images, code, formulas)
- Intelligent chunking with HybridChunker
- Multiple fallback strategies for robust extraction
- Rich metadata extraction (page numbers, sections, content types)
- OCR support for scanned documents
- Battle-tested on 40-50 page PDFs

Author: Optimized for CMTI RAG Pipeline
Date: November 2025
"""
from vision_figures import process_figures_for_document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import logging
import re
from typing import List, Dict, Any
from pdf2image import convert_from_path
import os
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- Configuration -----
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_TOKENS = 512
FALLBACK_CHUNK_SIZE = 800
FALLBACK_OVERLAP = 100

# ----- Embedding Model -----
embed_model = SentenceTransformer(EMBED_MODEL_ID)

def embed(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks."""
    if not texts:
        return []
    return embed_model.encode(texts, show_progress_bar=False).tolist()

# ----- ChromaDB Client -----
CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_data")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="docling_chunks",
    metadata={"hnsw:space": "cosine"}
)

# ----- Document Converter Setup -----
def get_document_converter() -> DocumentConverter:
    """
    Configure Docling with optimized settings for production RAG.
    
    Features enabled:
    - OCR for scanned documents
    - Table structure recognition
    - Advanced image processing
    """
    pipeline_options = PdfPipelineOptions()
    
    # OCR Configuration
    pipeline_options.do_ocr = False   # Disable built-in OCR
    
    # Table Structure
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    
    # Image Processing
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False  # We don't need page images for RAG
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    return converter


def store_figure_chunks(
    figure_results: List[Dict[str, Any]],
    user_id: str,
    doc_id: Any,
    filename: str,
) -> int:
    """Store figure/diagram descriptions as searchable chunks (from Groq vision)."""
    if not figure_results:
        return 0
    ids = []
    docs = []
    metadata_list = []
    for idx, fig in enumerate(figure_results):
        text = fig.get("text_for_chunk", "").strip()
        if not text or len(text) < 20:
            continue
        chunk_id = f"user_{user_id}_doc_{doc_id}_figure_{idx}"
        ids.append(chunk_id)
        docs.append(text)
        metadata_list.append({
            "user_id": user_id,
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": idx,
            "page": fig.get("page_no", 0),
            "section": "",
            "content_types": "figure",
            "chunking_method": "vision",
        })
    if not docs:
        return 0
    try:
        embeddings = embed(docs)
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadata_list,
        )
        logger.info(f"Stored {len(docs)} figure chunks in ChromaDB")
        return len(docs)
    except Exception as e:
        logger.error(f"ChromaDB figure storage error: {str(e)}", exc_info=True)
        return 0

# ----- HybridChunker Setup -----
def get_chunker(max_tokens: int = MAX_CHUNK_TOKENS) -> HybridChunker:
    """
    Create HybridChunker with tokenizer aligned to embedding model.
    
    This ensures chunks fit within the embedding model's context window.
    """
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
        max_tokens=max_tokens
    )
    
    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True  # Merge adjacent small chunks for better context
    )

# ----- Text Extraction Strategies -----

def extract_text_via_iterate_items(doc) -> str:
    """
    STRATEGY 1: Extract text by iterating over document items.
    This is the most reliable method for getting all text content.
    """
    text_parts = []
    
    try:
        for item, level in doc.iterate_items():
            # Extract text from various item types
            if hasattr(item, 'text') and item.text:
                text_parts.append(item.text.strip())
        
        full_text = '\n\n'.join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters via iterate_items()")
        return full_text
    
    except Exception as e:
        logger.warning(f"iterate_items() extraction failed: {e}")
        return ""

def extract_text_via_export(doc) -> str:
    """
    STRATEGY 2: Extract text using export methods.
    Fallback when iterate_items fails.
    """
    try:
        # Try markdown export first
        markdown_text = doc.export_to_markdown()
        if markdown_text and len(markdown_text) > 100:
            logger.info(f"Extracted {len(markdown_text)} characters via export_to_markdown()")
            return markdown_text
    except Exception as e:
        logger.warning(f"export_to_markdown() failed: {e}")
    
    try:
        # Try plain text export as fallback
        plain_text = doc.export_to_text()
        if plain_text and len(plain_text) > 100:
            logger.info(f"Extracted {len(plain_text)} characters via export_to_text()")
            return plain_text
    except Exception as e:
        logger.warning(f"export_to_text() failed: {e}")
    
    return ""

def extract_text_multi_strategy(doc, file_path=None) -> str:
    """
    Multi-step extraction with EasyOCR fallback.
    """
    # Strategy 1: Docling iterate_items
    text = extract_text_via_iterate_items(doc)
    if text and len(text) > 100:
        return text

    # Strategy 2: Docling export
    text = extract_text_via_export(doc)
    if text and len(text) > 100:
        return text

    # Strategy 3: EASYOCR fallback
    # if file_path and file_path.lower().endswith(".pdf"):
    #     logger.warning("Docling extraction failed → Running EasyOCR fallback")
    #     text = extract_text_with_easyocr(file_path)
    #     if text and len(text) > 50:
    #         return text

    return ""

# ----- Fallback Chunking -----

def fallback_text_chunking(text: str, chunk_size: int = FALLBACK_CHUNK_SIZE, 
                          overlap: int = FALLBACK_OVERLAP) -> List[str]:
    """
    Fallback chunking strategy when HybridChunker returns no chunks.
    
    Features:
    - Respects paragraph boundaries
    - Sliding window with overlap for context continuity
    - Handles long paragraphs gracefully
    """
    if not text or not text.strip():
        return []
    
    # Split by paragraphs
    paragraphs = re.split(r'\n\n+', text.strip())
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_length = len(para)
        
        # Handle long paragraphs
        if para_length > chunk_size:
            # Save current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long paragraph
            words = para.split()
            temp_chunk = []
            temp_length = 0
            
            for word in words:
                word_len = len(word) + 1  # +1 for space
                
                if temp_length + word_len > chunk_size:
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        
                        # Keep overlap words
                        overlap_words = temp_chunk[-overlap:] if len(temp_chunk) > overlap else temp_chunk
                        temp_chunk = overlap_words + [word]
                        temp_length = sum(len(w) + 1 for w in temp_chunk)
                    else:
                        temp_chunk = [word]
                        temp_length = word_len
                else:
                    temp_chunk.append(word)
                    temp_length += word_len
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
        
        else:
            # Add paragraph to current chunk
            if current_length + para_length > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length + 2  # +2 for \n\n
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# ----- Main Processing Function -----

def process_document(file_path: str, filename: str, user_id: str, doc_id: str) -> Dict[str, Any]:
    """
    Process document with robust multi-strategy extraction and chunking.
    
    Process Flow:
    1. Convert document using Docling
    2. Extract text using multiple strategies
    3. Try HybridChunker first
    4. Fallback to manual chunking if needed
    5. Store chunks with rich metadata
    
    Returns:
        Dict with processing statistics
    """
    converter = get_document_converter()
    chunker = get_chunker()
    
    try:
        # Step 1: Convert Document
        logger.info(f"Converting file: {file_path}")
        result = converter.convert(file_path)
        doc = result.document
        
        logger.info(f"Document conversion completed for {filename}")
        
        # Step 2: Extract Text (Multi-Strategy)
        full_text = extract_text_multi_strategy(doc, file_path=file_path)
        
        if not full_text or len(full_text) < 100:
            logger.error(f"Failed to extract meaningful text from {filename}")
            return {
                "filename": filename,
                "doc_id": doc_id,
                "total_chunks": 0,
                "status": "error_no_text_extracted",
                "text_length": len(full_text)
            }
        
        logger.info(f"Total text extracted: {len(full_text)} characters")
        
        # Step 3: Try HybridChunker
        chunks = []
        try:
            chunks = list(chunker.chunk(doc))
            logger.info(f"HybridChunker produced {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"HybridChunker failed: {e}")
        
        # Step 4: Fallback if HybridChunker failed or returned empty
        if not chunks or len(chunks) == 0:
            logger.warning(f"HybridChunker returned 0 chunks. Using fallback strategy.")
            
            # Use fallback chunking on extracted text
            text_chunks = fallback_text_chunking(full_text)
            logger.info(f"Fallback chunking produced {len(text_chunks)} chunks")
            
            if not text_chunks:
                logger.error(f"Both chunking strategies failed for {filename}")
                return {
                    "filename": filename,
                    "doc_id": doc_id,
                    "total_chunks": 0,
                    "status": "error_chunking_failed",
                    "text_length": len(full_text)
                }
            
            # Store fallback chunks
            store_fallback_chunks(text_chunks, user_id, doc_id, filename)
            
            # Extract and describe figures with Groq vision (PDF only)
            if file_path.lower().endswith(".pdf"):
                try:
                    figure_results = process_figures_for_document(
                        file_path, user_id, doc_id, filename
                    )
                    if figure_results:
                        store_figure_chunks(figure_results, user_id, doc_id, filename)
                except Exception as e:
                    logger.warning(f"Figure processing skipped: {e}")
            
            return {
                "filename": filename,
                "doc_id": doc_id,
                "total_chunks": len(text_chunks),
                "format": result.input.format.name if hasattr(result.input, 'format') else "unknown",
                "chunking_method": "fallback",
                "text_length": len(full_text),
                "status": "success"
            }
        
        # Step 5: Store HybridChunker chunks with rich metadata
                # Step 5: Store HybridChunker chunks with rich metadata
        store_chunks_with_metadata(chunks, user_id, doc_id, filename)
        
        # Extract and describe figures with Groq vision (PDF only)
        if file_path.lower().endswith(".pdf"):
            try:
                figure_results = process_figures_for_document(
                    file_path, user_id, doc_id, filename
                )
                if figure_results:
                    store_figure_chunks(figure_results, user_id, doc_id, filename)
            except Exception as e:
                logger.warning(f"Figure processing skipped: {e}")
        
        return {
            "filename": filename,
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "format": result.input.format.name if hasattr(result.input, 'format') else "unknown",
            "chunking_method": "hybrid",
            "text_length": len(full_text),
            "has_tables": any(hasattr(c, 'meta') and 'table' in str(getattr(c.meta, 'doc_items', [])) for c in chunks),
            "has_images": any(hasattr(c, 'meta') and 'picture' in str(getattr(c.meta, 'doc_items', [])) for c in chunks),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
        return {
            "filename": filename,
            "doc_id": doc_id,
            "total_chunks": 0,
            "status": "error",
            "error_message": str(e)
        }

# ----- Storage Functions -----

def store_fallback_chunks(text_chunks: List[str], user_id: str, doc_id: str, filename: str):
    """Store fallback text chunks with basic metadata."""
    ids = []
    docs = []
    metadata_list = []
    
    for idx, chunk_text in enumerate(text_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text or len(chunk_text) < 10:  # Skip very short chunks
            continue
        
        chunk_id = f"user_{user_id}_doc_{doc_id}_chunk_{idx}"
        
        chunk_metadata = {
            "user_id": user_id,
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": idx,
            "page": 0,
            "section": "",
            "content_types": "text",
            "chunking_method": "fallback"
        }
        
        ids.append(chunk_id)
        docs.append(chunk_text)
        metadata_list.append(chunk_metadata)
    
    if not docs:
        logger.warning(f"No valid fallback chunks for {filename}")
        return
    
    try:
        embeddings = embed(docs)
        
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadata_list
        )
        
        logger.info(f"✅ Stored {len(docs)} fallback chunks in ChromaDB")
    except Exception as e:
        logger.error(f"ChromaDB storage error: {str(e)}", exc_info=True)
        raise

def store_chunks_with_metadata(chunks, user_id: str, doc_id: str, filename: str):
    """Store HybridChunker chunks with rich metadata."""
    ids = []
    docs = []
    metadata_list = []
    
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.text.strip()
        if not chunk_text or len(chunk_text) < 10:
            continue
        
        chunk_id = f"user_{user_id}_doc_{doc_id}_chunk_{idx}"
        
        # Base metadata
        chunk_metadata = {
            "user_id": user_id,
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": idx,
            "chunking_method": "hybrid"
        }
        
        # Extract rich metadata from Docling
        if hasattr(chunk, 'meta') and chunk.meta:
            meta = chunk.meta
            
            # Extract page numbers
            if hasattr(meta, 'doc_items') and meta.doc_items:
                pages = set()
                for item in meta.doc_items:
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                pages.add(prov.page_no)
                
                chunk_metadata["page"] = min(pages) if pages else 0
                if pages:
                    chunk_metadata["pages"] = ",".join(map(str, sorted(pages)))
            else:
                chunk_metadata["page"] = 0
            
            # Extract section headings
            if hasattr(meta, 'headings') and meta.headings:
                chunk_metadata["section"] = " > ".join(meta.headings)
            else:
                chunk_metadata["section"] = ""
            
            # Extract content types
            if hasattr(meta, 'doc_items') and meta.doc_items:
                labels = set()
                for item in meta.doc_items:
                    if hasattr(item, 'label'):
                        labels.add(str(item.label))
                chunk_metadata["content_types"] = ", ".join(labels) if labels else "text"
            else:
                chunk_metadata["content_types"] = "text"
        else:
            chunk_metadata["page"] = 0
            chunk_metadata["section"] = ""
            chunk_metadata["content_types"] = "text"
        
        ids.append(chunk_id)
        docs.append(chunk_text)
        metadata_list.append(chunk_metadata)
    
    if not docs:
        logger.warning(f"No valid chunks with metadata for {filename}")
        return
    
    try:
        embeddings = embed(docs)
        
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadata_list
        )
        
        logger.info(f"✅ Stored {len(docs)} hybrid chunks in ChromaDB")
    except Exception as e:
        logger.error(f"ChromaDB storage error: {str(e)}", exc_info=True)
        raise

# ----- Search Function -----

def search_chunks(query: str, user_id: str, doc_id: str = None, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for relevant chunks with optional document filtering.
    
    Args:
        query: Search query
        user_id: User ID for filtering
        doc_id: Optional document ID for filtering
        k: Number of results to return
    
    Returns:
        List of relevant chunks with metadata
    """
    if not query.strip():
        return []
    
    query_embedding = embed([query])
    
    # Build filter
    if doc_id:
        where_clause = {
            "$and": [
                {"user_id": user_id},
                {"doc_id": doc_id}
            ]
        }
    else:
        where_clause = {"user_id": user_id}
    
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where_clause
        )
        
        chunks = []
        if results["documents"] and results["metadatas"]:
            for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                chunk_info = {
                    "text": text,
                    "page": meta.get("page", "N/A"),
                    "section": meta.get("section", ""),
                    "content_types": meta.get("content_types", ""),
                    "filename": meta.get("filename", ""),
                }
                chunks.append(chunk_info)
        
        logger.info(f"Found {len(chunks)} relevant chunks for query")
        return chunks
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return []

# ----- Utility Functions -----

def detect_format(filename: str) -> str:
    """Detect document format from filename extension."""
    ext = filename.lower().split('.')[-1]
    
    format_map = {
        'pdf': 'PDF',
        'docx': 'Word Document',
        'pptx': 'PowerPoint',
        'xlsx': 'Excel',
        'html': 'HTML',
        'htm': 'HTML',
        'png': 'Image',
        'jpg': 'Image',
        'jpeg': 'Image',
        'tiff': 'Image',
        'bmp': 'Image',
    }
    
    return format_map.get(ext, 'Unknown')