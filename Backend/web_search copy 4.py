"""
Simplified Web Search Module with Crawl4AI - Windows Compatible (FIXED)
------------------------------------------------------------------------
Fixed the Windows subprocess issue with proper event loop handling
"""

import asyncio
import logging
import sys
from typing import List, Dict, Optional
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# ============================================================================
# CRITICAL FIX for Windows - Must be at the very top before any async code
# ============================================================================
if sys.platform == 'win32':
    # Set the event loop policy for Windows
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Crawl4AI imports
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
    print("✅ Crawl4AI loaded successfully")
except ImportError as e:
    CRAWL4AI_AVAILABLE = False
    print(f"⚠️  Crawl4AI not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Quality domains
QUALITY_DOMAINS = {
    'wikipedia.org', 'britannica.com', 'nature.com', 'science.org',
    'ncbi.nlm.nih.gov', 'stackoverflow.com', 'github.com'
}

BLOCKED_DOMAINS = {
    'pinterest.com', 'instagram.com', 'facebook.com', 'twitter.com'
}


class SearchEngine:
    """Simple DuckDuckGo search"""
    
    @staticmethod
    async def search(query: str, num_results: int = 6) -> List[Dict]:
        """Search using DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    query,
                    max_results=num_results * 2,
                    region='in-en'
                )
                
                for result in search_results:
                    url = result.get('href', result.get('link', ''))
                    
                    # Basic URL filtering
                    if not SearchEngine._is_valid_url(url):
                        continue
                    
                    results.append({
                        'title': result.get('title', ''),
                        'url': url,
                        'snippet': result.get('body', result.get('snippet', ''))
                    })
                    
                    if len(results) >= num_results:
                        break
            
            logger.info(f"✅ Found {len(results)} search results")
            return results
        
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Basic URL validation"""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc.replace('www.', '')
            
            # Block bad domains
            if any(blocked in domain for blocked in BLOCKED_DOMAINS):
                return False
            
            return True
        except:
            return False


class SimpleCrawler:
    """Simplified crawler - Windows Compatible"""
    
    def __init__(self):
        self.browser_config = None
        if CRAWL4AI_AVAILABLE:
            # Minimal browser config for Windows
            self.browser_config = BrowserConfig(
                headless=True,
                verbose=False
            )
    
    async def crawl_url(self, url: str) -> Optional[Dict]:
        """
        Crawl single URL using Crawl4AI
        Windows compatible with proper error handling
        """
        if not CRAWL4AI_AVAILABLE:
            return await self._fallback_scrape(url)
        
        try:
            logger.info(f"🕷️  Crawling with Crawl4AI: {url}")
            
            # Create crawler and crawl
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=50
                )
                
                result = await crawler.arun(url=url, config=run_config)
                
                if not result.success:
                    logger.warning(f"⚠️  Crawl4AI failed, using fallback: {url}")
                    return await self._fallback_scrape(url)
                
                # Extract text
                text = ""
                if hasattr(result, 'markdown'):
                    if hasattr(result.markdown, 'raw_markdown'):
                        text = result.markdown.raw_markdown
                    elif isinstance(result.markdown, str):
                        text = result.markdown
                
                if not text and result.html:
                    # Parse HTML if markdown not available
                    soup = BeautifulSoup(result.html, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                
                text = self._clean_text(text)
                
                if len(text) < 200:
                    logger.warning(f"⚠️  Content too short, using fallback: {url}")
                    return await self._fallback_scrape(url)
                
                logger.info(f"✅ Crawl4AI success {url}: {len(text)} chars")
                
                return {
                    'url': url,
                    'text': text,
                    'title': result.title or '',
                    'success': True,
                    'method': 'crawl4ai'
                }
        
        except NotImplementedError as e:
            # Windows subprocess issue - use fallback
            logger.warning(f"⚠️  Windows subprocess issue, using fallback for {url}")
            return await self._fallback_scrape(url)
        
        except Exception as e:
            logger.error(f"❌ Crawl4AI error for {url}: {str(e)[:100]}")
            return await self._fallback_scrape(url)
    
    async def _fallback_scrape(self, url: str) -> Optional[Dict]:
        """Simple fallback scraping"""
        try:
            logger.info(f"🌐 Fallback scraping: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return None
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove unwanted tags
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                        tag.decompose()
                    
                    # Get main content
                    main = soup.find('main') or soup.find('article') or soup.find('body')
                    if not main:
                        return None
                    
                    text = main.get_text(separator=' ', strip=True)
                    text = self._clean_text(text)
                    
                    if len(text) < 200:
                        return None
                    
                    logger.info(f"✅ Fallback success {url}: {len(text)} chars")
                    
                    return {
                        'url': url,
                        'text': text,
                        'title': soup.title.string if soup.title else '',
                        'success': True,
                        'method': 'fallback'
                    }
        
        except Exception as e:
            logger.error(f"❌ Fallback error for {url}: {str(e)[:100]}")
            return None
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        return text.strip()


def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """Simple text chunking"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size // 6):  # ~6 chars per word avg
        chunk = ' '.join(words[i:i + chunk_size // 6])
        if len(chunk) > 100:
            chunks.append(chunk)
    
    return chunks[:5]  # Max 5 chunks per source


async def search_and_crawl(query: str, num_results: int = 6) -> Dict:
    """
    Main function: Search and crawl
    Simplified - sequential processing
    """
    logger.info(f"🔍 Starting search for: {query}")
    
    # Step 1: Search
    search_results = await SearchEngine.search(query, num_results)
    
    if not search_results:
        return {
            'chunks': [],
            'sources': [],
            'embeddings': None,
            'error': 'No search results found'
        }
    
    # Step 2: Crawl each URL (sequentially)
    crawler = SimpleCrawler()
    all_chunks = []
    all_sources = []
    crawl_methods = {'crawl4ai': 0, 'fallback': 0}
    
    for result in search_results:
        crawl_result = await crawler.crawl_url(result['url'])
        
        if not crawl_result or not crawl_result.get('success'):
            continue
        
        # Track which method worked
        method = crawl_result.get('method', 'unknown')
        crawl_methods[method] = crawl_methods.get(method, 0) + 1
        
        # Chunk the content
        chunks = chunk_text(crawl_result['text'])
        
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_sources.append({
                'url': result['url'],
                'title': crawl_result['title'] or result['title'],
                'snippet': result['snippet'],
                'chunk_index': idx,
                'method': method
            })
    
    if not all_chunks:
        return {
            'chunks': [],
            'sources': [],
            'embeddings': None,
            'error': 'Failed to crawl any sources'
        }
    
    # Step 3: Generate embeddings
    logger.info(f"🧠 Generating embeddings for {len(all_chunks)} chunks")
    logger.info(f"📊 Crawl methods used: {crawl_methods}")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=False)
    
    logger.info(f"✅ Complete: {len(all_chunks)} chunks from {len(search_results)} sources")
    
    return {
        'chunks': all_chunks,
        'sources': all_sources,
        'embeddings': embeddings,
        'crawl_methods': crawl_methods
    }


def find_relevant_chunks(query: str, chunks: List[str], sources: List[Dict],
                         embeddings: np.ndarray, k: int = 5) -> List[Dict]:
    """Find most relevant chunks"""
    query_embedding = embedding_model.encode([query])[0]
    
    # Calculate similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top k
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': chunks[idx],
            'similarity': float(similarities[idx]),
            'source': sources[idx]
        })
    
    return results


# Test function
async def test_crawler():
    """Test the crawler"""
    print("\n" + "="*60)
    print("TESTING WEB SEARCH & CRAWL")
    print("="*60 + "\n")
    
    query = "how does the Replace works in the spark in python"
    result = await search_and_crawl(query, num_results=3)
    
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n✅ Success!")
    print(f"   Chunks: {len(result['chunks'])}")
    print(f"   Sources: {len(set(s['url'] for s in result['sources']))}")
    print(f"   Crawl methods: {result.get('crawl_methods', {})}")
    
    # Test relevance search
    if result['chunks']:
        relevant = find_relevant_chunks(
            query,
            result['chunks'],
            result['sources'],
            result['embeddings'],
            k=3
        )
        
        print(f"\n📊 Top 3 relevant chunks:")
        for i, r in enumerate(relevant, 1):
            print(f"\n{i}. Score: {r['similarity']:.3f}")
            print(f"   Source: {r['source']['url']}")
            print(f"   Method: {r['source']['method']}")
            print(f"   Text: {r['text'][:150]}...")


if __name__ == "__main__":
    # Create new event loop explicitly for Windows
    if sys.platform == 'win32':
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_crawler())
        loop.close()
    else:
        asyncio.run(test_crawler())