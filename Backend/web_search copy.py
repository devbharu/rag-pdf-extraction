"""
Web Search Module for RAG Pipeline
Simplified and fixed version with reliable web scraping
"""

import asyncio
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from urllib.parse import urlparse
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo (free, no API key required)
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of dicts with 'title', 'url', 'snippet'
    """
    try:
        logger.info(f"Searching DuckDuckGo for: {query}")
        results = []
        
        with DDGS() as ddgs:
            # Use text search with region for better results
            search_results = ddgs.text(
                query, 
                max_results=num_results * 2,  # Get more results to filter
                region='wt-wt',  # Worldwide
                safesearch='moderate'
            )
            
            for result in search_results:
                url = result.get('href', result.get('link', ''))
                
                # Skip low-quality domains
                if any(skip in url.lower() for skip in ['zhihu.com', 'quora.com', 'reddit.com/r/', 'pinterest.com']):
                    continue
                
                results.append({
                    'title': result.get('title', 'No title'),
                    'url': url,
                    'snippet': result.get('body', result.get('snippet', ''))
                })
                
                if len(results) >= num_results:
                    break
        
        logger.info(f"Found {len(results)} quality results")
        return results
        
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {str(e)}")
        return []


def scrape_url(url: str, timeout: int = 10) -> Optional[str]:
    """
    Scrape content from URL with proper error handling
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        Extracted text content or None if failed
    """
    try:
        logger.info(f"Scraping: {url}")
        
        # Improved headers to avoid blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Try to find main content
        main_content = None
        
        # Look for common main content containers
        for selector in ['main', 'article', '[role="main"]', '.main-content', '#main-content', '.content', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            logger.warning(f"No content found for {url}")
            return None
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        # Validate content quality
        if len(text) < 100:
            logger.warning(f"Content too short for {url}: {len(text)} chars")
            return None
        
        logger.info(f"Successfully scraped {url}: {len(text)} chars")
        return text
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {url}: {e}")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Timeout scraping {url}")
        return None
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {str(e)}")
        return None


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks with sentence awareness
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split by sentences first (simple approach)
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in '.!?' and len(current) > 20:
            sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    # Group sentences into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


async def process_web_search(query: str, num_results: int = 5) -> Dict:
    """
    Main orchestrator for web search functionality
    
    1. Search DuckDuckGo for relevant URLs
    2. Scrape top results (with retry logic)
    3. Extract and chunk content
    4. Generate embeddings
    5. Return structured data
    
    Args:
        query: User's search query
        num_results: Number of search results to process
        
    Returns:
        Dict with 'chunks', 'sources', and 'embeddings'
    """
    logger.info(f"Processing web search for: {query}")
    
    # Step 1: Search the web
    search_results = search_web(query, num_results)
    
    if not search_results:
        return {
            'chunks': [],
            'sources': [],
            'embeddings': None,
            'error': 'No search results found'
        }
    
    # Step 2: Scrape content from URLs
    all_chunks = []
    all_sources = []
    successful_scrapes = 0
    
    for result in search_results:
        url = result['url']
        title = result['title']
        snippet = result['snippet']
        
        # Scrape content
        content = scrape_url(url, timeout=15)
        
        if not content:
            logger.warning(f"Skipping {url}: failed to scrape")
            continue
        
        successful_scrapes += 1
        
        # Chunk the content
        chunks = chunk_text(content, chunk_size=700, overlap=100)
        
        # Take first 3 chunks (usually most relevant)
        for i, chunk in enumerate(chunks[:3]):
            all_chunks.append(chunk)
            all_sources.append({
                'url': url,
                'title': title,
                'snippet': snippet,
                'chunk_index': i,
                'domain': urlparse(url).netloc
            })
        
        # Small delay to be respectful
        time.sleep(0.5)
    
    if not all_chunks:
        return {
            'chunks': [],
            'sources': [],
            'embeddings': None,
            'error': 'Failed to extract content from any source'
        }
    
    # Step 3: Generate embeddings for semantic search
    logger.info(f"Generating embeddings for {len(all_chunks)} chunks from {successful_scrapes} sources")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=False)
    
    return {
        'chunks': all_chunks,
        'sources': all_sources,
        'embeddings': embeddings,
        'total_sources': len(search_results),
        'successful_crawls': successful_scrapes
    }


def find_relevant_chunks(query: str, chunks: List[str], sources: List[Dict], 
                        embeddings: np.ndarray, k: int = 5) -> List[Dict]:
    """
    Find most relevant chunks using semantic similarity
    
    Args:
        query: User's query
        chunks: List of text chunks
        sources: List of source metadata
        embeddings: Precomputed embeddings for chunks
        k: Number of top chunks to return
        
    Returns:
        List of dicts with chunk text and source info
    """
    # Encode query
    query_embedding = embedding_model.encode([query], show_progress_bar=False)[0]
    
    # Calculate cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # Build results
    results = []
    seen_urls = set()
    
    for idx in top_indices:
        source = sources[idx]
        url = source['url']
        
        # Prefer diversity - limit chunks per source
        if url in seen_urls:
            continue
        
        results.append({
            'text': chunks[idx],
            'similarity': float(similarities[idx]),
            'source': source
        })
        
        seen_urls.add(url)
    
    return results