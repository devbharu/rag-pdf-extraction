"""
Enhanced Web Search Module with Crawl4AI
-----------------------------------------
Features:
- Multiple search engines (Google, Brave, DuckDuckGo)
- Advanced crawling with Crawl4AI
- Language and region filtering
- Parallel async processing
- Quality content validation
- Semantic ranking
"""

import asyncio
import logging
from typing import List, Dict, Optional
from urllib.parse import urlparse, quote_plus
import aiohttp
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os

# Crawl4AI imports
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    print("⚠️  Crawl4AI not installed. Install with: pip install crawl4ai")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Quality domain list (prefer these)
QUALITY_DOMAINS = {
    'wikipedia.org', 'britannica.com', 'nature.com', 'science.org',
    'ncbi.nlm.nih.gov', 'arxiv.org', 'sciencedirect.com', 'springer.com',
    'ieee.org', 'acm.org', 'stackoverflow.com', 'github.com', 'medium.com'
}

# Blocked domains (avoid these)
BLOCKED_DOMAINS = {
    'pinterest.com', 'instagram.com', 'facebook.com', 'twitter.com',
    'tiktok.com', 'reddit.com', 'quora.com'
}

# Non-English domain patterns to filter
NON_ENGLISH_PATTERNS = [
    r'\.cn$', r'\.jp$', r'\.kr$', r'\.tw$', r'\.ru$',
    r'\.ar$', r'\.sa$', r'\.ae$', r'\.in/hindi', r'\.in/tamil'
]


class SearchEngine:
    """Multi-engine search with intelligent fallback"""
    
    @staticmethod
    async def google_search(query: str, num_results: int = 10) -> List[Dict]:
        """
        Search using Google Custom Search API (requires API key)
        Best quality but needs setup
        """
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not api_key or not search_engine_id:
            logger.warning("Google API not configured, skipping")
            return []
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': query,
            'num': min(num_results, 10),
            'lr': 'lang_en'  # English only
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get('items', []):
                            results.append({
                                'title': item.get('title', ''),
                                'url': item.get('link', ''),
                                'snippet': item.get('snippet', ''),
                                'source': 'google'
                            })
                        logger.info(f"Google found {len(results)} results")
                        return results
        except Exception as e:
            logger.error(f"Google search error: {e}")
        
        return []
    
    @staticmethod
    async def brave_search(query: str, num_results: int = 10) -> List[Dict]:
        """
        Search using Brave Search API (free tier available)
        Good balance of quality and accessibility
        """
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        
        if not api_key:
            logger.warning("Brave API not configured, skipping")
            return []
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            'Accept': 'application/json',
            'X-Subscription-Token': api_key
        }
        params = {
            'q': query,
            'count': num_results,
            'search_lang': 'en',
            'country': 'US'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get('web', {}).get('results', []):
                            results.append({
                                'title': item.get('title', ''),
                                'url': item.get('url', ''),
                                'snippet': item.get('description', ''),
                                'source': 'brave'
                            })
                        logger.info(f"Brave found {len(results)} results")
                        return results
        except Exception as e:
            logger.error(f"Brave search error: {e}")
        
        return []
    
    @staticmethod
    async def duckduckgo_search(query: str, num_results: int = 10) -> List[Dict]:
        """
        Fallback to DuckDuckGo with better filtering
        """
        try:
            # Use updated package if available
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    query,
                    max_results=num_results * 3,  # Get extras for filtering
                    region='us-en',  # US English results
                    safesearch='moderate'
                )
                
                for result in search_results:
                    url = result.get('href', result.get('link', ''))
                    
                    # Apply filters
                    if not SearchEngine._is_quality_url(url):
                        continue
                    
                    results.append({
                        'title': result.get('title', ''),
                        'url': url,
                        'snippet': result.get('body', result.get('snippet', '')),
                        'source': 'duckduckgo'
                    })
                    
                    if len(results) >= num_results:
                        break
            
            logger.info(f"DuckDuckGo found {len(results)} quality results")
            return results
        
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    @staticmethod
    def _is_quality_url(url: str) -> bool:
        """Filter out low-quality and non-English URLs"""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc.replace('www.', '')
            
            # Block bad domains
            if any(blocked in domain for blocked in BLOCKED_DOMAINS):
                return False
            
            # Block non-English TLDs
            if any(re.search(pattern, url) for pattern in NON_ENGLISH_PATTERNS):
                return False
            
            # Check for Chinese/Japanese/Korean characters in URL
            if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', url):
                return False
            
            return True
        
        except Exception:
            return False


class SmartCrawler:
    """Advanced web crawling with Crawl4AI"""
    
    def __init__(self):
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
        )
    
    async def crawl_url(self, url: str) -> Optional[Dict]:
        """
        Crawl URL using Crawl4AI with intelligent content extraction
        """
        if not CRAWL4AI_AVAILABLE:
            # Fallback to basic scraping
            return await self._basic_scrape(url)
        
        try:
            logger.info(f"🕷️  Crawling: {url}")
            
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=50,  # Min words for quality
                    exclude_external_links=True,
                    process_iframes=False,
                    remove_overlay_elements=True,
                    wait_until="networkidle"
                )
                
                result = await crawler.arun(url=url, config=run_config)
                
                if not result.success:
                    logger.warning(f"❌ Crawl failed: {url}")
                    return None
                
                # Extract clean text
                text = result.markdown or result.cleaned_html or result.html
                text = self._clean_text(text)
                
                # Validate content quality
                if not self._is_quality_content(text):
                    logger.warning(f"⚠️  Low quality content: {url}")
                    return None
                
                logger.info(f"✅ Successfully crawled {url}: {len(text)} chars")
                
                return {
                    'url': url,
                    'text': text,
                    'title': result.title or '',
                    'success': True
                }
        
        except Exception as e:
            logger.error(f"Crawl4AI error for {url}: {e}")
            return await self._basic_scrape(url)
    
    async def _basic_scrape(self, url: str) -> Optional[Dict]:
        """Fallback basic scraping"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status != 200:
                        return None
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove unwanted elements
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                        tag.decompose()
                    
                    # Extract main content
                    main = soup.find('main') or soup.find('article') or soup.find('body')
                    if not main:
                        return None
                    
                    text = main.get_text(separator='\n', strip=True)
                    text = self._clean_text(text)
                    
                    if not self._is_quality_content(text):
                        return None
                    
                    return {
                        'url': url,
                        'text': text,
                        'title': soup.title.string if soup.title else '',
                        'success': True
                    }
        
        except Exception as e:
            logger.error(f"Basic scrape error for {url}: {e}")
            return None
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
        return text.strip()
    
    @staticmethod
    def _is_quality_content(text: str) -> bool:
        """Validate content quality"""
        if len(text) < 200:
            return False
        
        # Check for minimum English content
        english_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
        if english_ratio < 0.5:
            return False
        
        # Check word diversity
        words = text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.3:  # Too repetitive
            return False
        
        return True


def chunk_text_semantic(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Smart text chunking with sentence boundaries
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [c for c in chunks if len(c) > 100]  # Filter short chunks


async def process_web_search(query: str, num_results: int = 6) -> Dict:
    """
    Main orchestrator with multi-engine search and advanced crawling
    """
    logger.info(f"🔍 Processing search: {query}")
    
    # Step 1: Multi-engine search with fallback
    all_results = []
    
    # Try Google first (best quality)
    google_results = await SearchEngine.google_search(query, num_results)
    all_results.extend(google_results)
    
    # Try Brave if needed
    if len(all_results) < num_results:
        brave_results = await SearchEngine.brave_search(query, num_results)
        all_results.extend(brave_results)
    
    # Fallback to DuckDuckGo
    if len(all_results) < num_results:
        ddg_results = await SearchEngine.duckduckgo_search(query, num_results)
        all_results.extend(ddg_results)
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for result in all_results:
        if result['url'] not in seen_urls:
            seen_urls.add(result['url'])
            unique_results.append(result)
    
    search_results = unique_results[:num_results]
    
    if not search_results:
        return {
            'chunks': [],
            'sources': [],
            'embeddings': None,
            'error': 'No search results found'
        }
    
    logger.info(f"📊 Found {len(search_results)} unique results")
    
    # Step 2: Parallel crawling with Crawl4AI
    crawler = SmartCrawler()
    crawl_tasks = [crawler.crawl_url(result['url']) for result in search_results]
    crawled_results = await asyncio.gather(*crawl_tasks)
    
    # Step 3: Process successful crawls
    all_chunks = []
    all_sources = []
    successful_crawls = 0
    
    for i, (search_result, crawl_result) in enumerate(zip(search_results, crawled_results)):
        if not crawl_result or not crawl_result.get('success'):
            continue
        
        successful_crawls += 1
        
        # Chunk the content
        chunks = chunk_text_semantic(crawl_result['text'], chunk_size=800, overlap=100)
        
        # Take top chunks
        for j, chunk in enumerate(chunks[:3]):
            all_chunks.append(chunk)
            all_sources.append({
                'url': search_result['url'],
                'title': crawl_result.get('title', search_result['title']),
                'snippet': search_result.get('snippet', ''),
                'chunk_index': j,
                'domain': urlparse(search_result['url']).netloc,
                'source_engine': search_result.get('source', 'unknown')
            })
    
    if not all_chunks:
        return {
            'chunks': [],
            'sources': [],
            'embeddings': None,
            'error': 'Failed to extract content from any source'
        }
    
    # Step 4: Generate embeddings
    logger.info(f"🧠 Generating embeddings for {len(all_chunks)} chunks from {successful_crawls} sources")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=False)
    
    logger.info(f"✅ Search complete: {successful_crawls}/{len(search_results)} sources crawled")
    
    return {
        'chunks': all_chunks,
        'sources': all_sources,
        'embeddings': embeddings,
        'total_sources': len(search_results),
        'successful_crawls': successful_crawls
    }


def find_relevant_chunks(query: str, chunks: List[str], sources: List[Dict],
                         embeddings: np.ndarray, k: int = 5) -> List[Dict]:
    """
    Find most relevant chunks using semantic similarity
    """
    # Encode query
    query_embedding = embedding_model.encode([query], show_progress_bar=False)[0]
    
    # Calculate cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:k * 2]  # Get extras for diversity
    
    # Build results with diversity
    results = []
    seen_domains = set()
    
    for idx in top_indices:
        source = sources[idx]
        domain = source['domain']
        
        # Prefer domain diversity
        if domain in seen_domains and len(results) >= k:
            continue
        
        results.append({
            'text': chunks[idx],
            'similarity': float(similarities[idx]),
            'source': source
        })
        
        seen_domains.add(domain)
        
        if len(results) >= k:
            break
    
    return results


# Export main functions
__all__ = ['process_web_search', 'find_relevant_chunks']