"""
Web Search Module - Premium Search APIs + Smart Crawling
Uses Tavily (primary) → Exa AI (fallback) → Jina AI (enhancement)
"""

import asyncio
import logging
import os
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse
import aiohttp
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Premium Search APIs
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("⚠️  Tavily not available")

try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    print("⚠️  Exa AI not available")

# Content extraction fallbacks
try:
    from trafilatura import fetch_url, extract
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model (singleton pattern)
_embedding_model = None

def get_embedding_model():
    """Lazy load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

# Quality and blocked domains
BLOCKED_DOMAINS = {
    'pinterest.com', 'instagram.com', 'facebook.com', 'twitter.com'
}


class TavilySearchEngine:
    """
    Tavily Search API - Primary search engine
    Provides high-quality results with pre-extracted content
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("Tavily API key not found")
        self.client = TavilyClient(api_key=self.api_key)
    
    async def search(self, query: str, num_results: int = 6) -> Dict:
        """
        Search using Tavily API
        
        Returns:
            Dict with 'results' (list of search results) and 'answer' (AI summary)
        """
        try:
            logger.info(f"🔵 Tavily search: {query}")
            
            # Tavily is synchronous, run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    query=query,
                    max_results=num_results,
                    include_answer=True,
                    include_raw_content=False
                )
            )
            
            results = []
            for item in response.get('results', []):
                url = item.get('url', '')
                
                # Filter blocked domains
                if self._is_blocked_url(url):
                    continue
                
                results.append({
                    'url': url,
                    'title': item.get('title', ''),
                    'content': item.get('content', ''),  # Pre-extracted content!
                    'score': item.get('score', 0.0),
                    'favicon': item.get('favicon', ''),
                    'source': 'tavily'
                })
            
            logger.info(f"✅ Tavily found {len(results)} results")
            
            return {
                'results': results,
                'answer': response.get('answer', ''),
                'success': True
            }
        
        except Exception as e:
            logger.error(f"❌ Tavily error: {str(e)[:100]}")
            return {'results': [], 'answer': '', 'success': False, 'error': str(e)}
    
    @staticmethod
    def _is_blocked_url(url: str) -> bool:
        """Check if URL is from blocked domain"""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc.replace('www.', '')
            return any(blocked in domain for blocked in BLOCKED_DOMAINS)
        except:
            return False


class ExaSearchEngine:
    """
    Exa AI Search API - Fallback search engine
    Provides neural search with full text content extraction
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('EXA_API_KEY')
        if not self.api_key:
            raise ValueError("Exa API key not found")
        self.client = Exa(api_key=self.api_key)
    
    async def search(self, query: str, num_results: int = 6) -> Dict:
        """
        Search using Exa AI API with content extraction
        
        Returns:
            Dict with 'results' (list of search results with full text)
        """
        try:
            logger.info(f"🟡 Exa AI search: {query}")
            
            # Exa is synchronous, run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.search_and_contents(
                    query=query,
                    text=True,
                    type="auto",
                    num_results=num_results
                )
            )
            
            results = []
            for item in response.results:
                url = item.url
                
                # Filter blocked domains
                if self._is_blocked_url(url):
                    continue
                
                # Exa provides full text content
                text = item.text if hasattr(item, 'text') else ''
                
                results.append({
                    'url': url,
                    'title': item.title if hasattr(item, 'title') else '',
                    'content': text,  # Full text content!
                    'author': item.author if hasattr(item, 'author') else '',
                    'published_date': item.published_date if hasattr(item, 'published_date') else '',
                    'favicon': item.favicon if hasattr(item, 'favicon') else '',
                    'score': 0.95,  # Exa doesn't provide scores, use high default
                    'source': 'exa'
                })
            
            logger.info(f"✅ Exa AI found {len(results)} results")
            
            return {
                'results': results,
                'success': True
            }
        
        except Exception as e: 
            logger.error(f"❌ Exa AI error: {str(e)[:100]}")
            return {'results': [], 'success': False, 'error': str(e)}
    
    @staticmethod
    def _is_blocked_url(url: str) -> bool:
        """Check if URL is from blocked domain"""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc.replace('www.', '')
            return any(blocked in domain for blocked in BLOCKED_DOMAINS)
        except:
            return False


class ContentEnhancer:
    """
    Enhance content using Jina AI, Trafilatura, or Playwright
    Only used when search APIs don't provide enough content
    """
    
    def __init__(self, jina_api_key: Optional[str] = None):
        self.jina_api_key = jina_api_key
        self.jina_endpoint = "https://r.jina.ai/"
        self.stats = {
            'jina': 0,
            'trafilatura': 0,
            'playwright': 0,
            'skipped': 0
        }
    
    async def enhance_if_needed(self, url: str, existing_content: str, min_length: int = 300) -> Optional[str]:
        """
        Enhance content only if existing content is insufficient
        
        Args:
            url: URL to fetch content from
            existing_content: Content already provided by search API
            min_length: Minimum content length to skip enhancement
            
        Returns:
            Enhanced content or existing content if sufficient
        """
        # If we already have good content, skip enhancement
        if len(existing_content) >= min_length:
            self.stats['skipped'] += 1
            return existing_content
        
        logger.info(f"📝 Enhancing content for: {url}")
        
        # Try enhancement methods
        enhanced = await self._enhance_with_jina(url)
        if enhanced and len(enhanced) > len(existing_content):
            self.stats['jina'] += 1
            return enhanced
        
        if TRAFILATURA_AVAILABLE:
            enhanced = await self._enhance_with_trafilatura(url)
            if enhanced and len(enhanced) > len(existing_content):
                self.stats['trafilatura'] += 1
                return enhanced
        
        if PLAYWRIGHT_AVAILABLE:
            enhanced = await self._enhance_with_playwright(url)
            if enhanced and len(enhanced) > len(existing_content):
                self.stats['playwright'] += 1
                return enhanced
        
        # Return existing content if enhancement failed
        return existing_content
    
    async def _enhance_with_jina(self, url: str) -> Optional[str]:
        """Enhance using Jina AI Reader API"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            if self.jina_api_key:
                headers['Authorization'] = f'Bearer {self.jina_api_key}'
            
            jina_url = f"{self.jina_endpoint}{url}"
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(jina_url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._clean_text(content)
        except:
            pass
        return None
    
    async def _enhance_with_trafilatura(self, url: str) -> Optional[str]:
        """Enhance using Trafilatura"""
        try:
            loop = asyncio.get_event_loop()
            downloaded = await loop.run_in_executor(None, fetch_url, url)
            if downloaded:
                text = await loop.run_in_executor(
                    None,
                    lambda: extract(downloaded, include_comments=False, include_tables=True)
                )
                if text:
                    return self._clean_text(text)
        except:
            pass
        return None
    
    async def _enhance_with_playwright(self, url: str) -> Optional[str]:
        """Enhance using Playwright (for complex JS sites)"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until='networkidle', timeout=10000)
                content = await page.content()
                await browser.close()
                
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                    tag.decompose()
                
                main = soup.find('main') or soup.find('article') or soup.find('body')
                if main:
                    text = main.get_text(separator=' ', strip=True)
                    return self._clean_text(text)
        except:
            pass
        return None
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        return text.strip()
    
    def get_stats(self) -> Dict:
        """Get enhancement statistics"""
        return self.stats.copy()


def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """Simple text chunking"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size // 6):
        chunk = ' '.join(words[i:i + chunk_size // 6])
        if len(chunk) > 100:
            chunks.append(chunk)
    
    return chunks[:5]  # Max 5 chunks per source


async def search_and_process(
    query: str,
    num_results: int = 6,
    tavily_api_key: Optional[str] = None,
    exa_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
    enhance_content: bool = False
) -> Dict:
    """
    Main function: Search using premium APIs and process results
    
    Args:
        query: Search query
        num_results: Number of results to fetch
        tavily_api_key: Tavily API key (optional, uses env var)
        exa_api_key: Exa API key (optional, uses env var)
        jina_api_key: Jina API key for content enhancement (optional)
        enhance_content: Whether to enhance content with additional crawling
        
    Returns:
        Dict with chunks, sources, embeddings, and stats
    """
    logger.info(f"🔍 Starting search for: {query}")
    
    search_results = []
    ai_answer = ""
    search_method = ""
    
    # Try Tavily first (primary)
    if TAVILY_AVAILABLE:
        try:
            tavily = TavilySearchEngine(api_key=tavily_api_key)
            response = await tavily.search(query, num_results)
            
            if response.get('success') and response.get('results'):
                search_results = response['results']
                ai_answer = response.get('answer', '')
                search_method = 'tavily'
                logger.info(f"✅ Using Tavily results")
        except Exception as e:
            logger.warning(f"⚠️  Tavily failed: {str(e)[:100]}")
    
    # Fallback to Exa AI if Tavily failed
    if not search_results and EXA_AVAILABLE:
        try:
            exa = ExaSearchEngine(api_key=exa_api_key)
            response = await exa.search(query, num_results)
            
            if response.get('success') and response.get('results'):
                search_results = response['results']
                search_method = 'exa'
                logger.info(f"✅ Using Exa AI results")
        except Exception as e:
            logger.warning(f"⚠️  Exa AI failed: {str(e)[:100]}")
    
    if not search_results:
        return {
            'chunks': [],
            'sources': [],
            'embeddings': None,
            'error': 'No search results found from any API',
            'search_method': 'none'
        }
    
    # Process results and optionally enhance content
    enhancer = ContentEnhancer(jina_api_key=jina_api_key) if enhance_content else None
    all_chunks = []
    all_sources = []
    
    for result in search_results:
        content = result.get('content', '')
        
        # Optionally enhance content if it's too short
        if enhancer and len(content) < 300:
            content = await enhancer.enhance_if_needed(result['url'], content)
        
        if len(content) < 100:
            logger.warning(f"⚠️  Skipping {result['url']}: content too short")
            continue
        
        # Chunk the content
        chunks = chunk_text(content)
        
        # Extract domain
        try:
            domain = urlparse(result['url']).netloc.replace('www.', '')
        except:
            domain = 'unknown'
        
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_sources.append({
                'url': result['url'],
                'title': result.get('title', ''),
                'domain': domain,
                'score': result.get('score', 0.0),
                'favicon': result.get('favicon', ''),
                'author': result.get('author', ''),
                'published_date': result.get('published_date', ''),
                'chunk_index': idx,
                'source_api': result.get('source', search_method),
                'snippet': chunk[:200]
            })
    
    if not all_chunks:
        return {
            'chunks': [],
            'sources': [],
            'embeddings': None,
            'error': 'Failed to extract content from search results',
            'search_method': search_method,
            'total_results': len(search_results)
        }
    
    # Generate embeddings
    logger.info(f"🧠 Generating embeddings for {len(all_chunks)} chunks")
    
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=False)
    
    result_dict = {
        'chunks': all_chunks,
        'sources': all_sources,
        'embeddings': embeddings,
        'search_method': search_method,
        'ai_answer': ai_answer,
        'total_results': len(search_results),
        'successful_chunks': len(all_chunks)
    }
    
    if enhancer:
        result_dict['enhancement_stats'] = enhancer.get_stats()
    
    logger.info(f"✅ Complete: {len(all_chunks)} chunks from {len(search_results)} sources via {search_method}")
    
    return result_dict


def find_relevant_chunks(query: str, chunks: List[str], sources: List[Dict],
                         embeddings: np.ndarray, k: int = 5) -> List[Dict]:
    """Find most relevant chunks using semantic similarity"""
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([query])[0]
    
    # Calculate cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': chunks[idx],
            'similarity': float(similarities[idx]),
            'source': sources[idx]
        })
    
    return results


# ============================================================================
# ROUTER-COMPATIBLE FUNCTION
# ============================================================================

async def process_web_search(
    query: str,
    num_results: int = 6,
    tavily_api_key: Optional[str] = None,
    exa_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
    enhance_content: bool = False
) -> Dict:
    """
    FastAPI router-compatible function for web search.
    
    Args:
        query: The search query string
        num_results: Number of search results to process (default 6)
        tavily_api_key: Tavily API key (optional)
        exa_api_key: Exa API key (optional)
        jina_api_key: Jina API key for content enhancement (optional)
        enhance_content: Whether to enhance content with additional crawling
    
    Returns:
        Dict with keys:
            - chunks: List of text chunks
            - sources: List of source metadata dicts
            - embeddings: numpy array of embeddings
            - search_method: Which API was used ('tavily' or 'exa')
            - ai_answer: AI-generated answer (from Tavily)
            - error: Optional error message
    """
    try:
        result = await search_and_process(
            query,
            num_results,
            tavily_api_key,
            exa_api_key,
            jina_api_key,
            enhance_content
        )
        return result
    except Exception as e:
        logger.error(f"❌ process_web_search error: {str(e)}")
        return {
            'chunks': [],
            'sources': [],
            'embeddings': None,
            'error': f'Search processing failed: {str(e)}',
            'search_method': 'none'
        }


# ============================================================================
# Test function
# ============================================================================

async def test_search():
    """Test the premium search APIs"""
    print("\n" + "="*60)
    print("TESTING PREMIUM SEARCH APIs (Tavily + Exa AI)")
    print("="*60 + "\n")
    
    query = "how does perplexity AI work"
    result = await process_web_search(query, num_results=5, enhance_content=False)
    
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n✅ Success!")
    print(f"   Search method: {result.get('search_method', 'unknown')}")
    print(f"   Total results: {result.get('total_results', 0)}")
    print(f"   Chunks: {len(result['chunks'])}")
    print(f"   Sources: {len(set(s['url'] for s in result['sources']))}")
    
    if result.get('ai_answer'):
        print(f"\n🤖 AI Answer (from Tavily):")
        print(f"   {result['ai_answer'][:200]}...")
    
    if result.get('enhancement_stats'):
        print(f"\n📊 Enhancement stats: {result['enhancement_stats']}")
    
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
            print(f"\n{i}. Similarity: {r['similarity']:.3f} | Score: {r['source'].get('score', 0):.3f}")
            print(f"   Source: {r['source']['title']}")
            print(f"   URL: {r['source']['url']}")
            print(f"   API: {r['source'].get('source_api', 'unknown')}")
            print(f"   Text: {r['text'][:150]}...")


if __name__ == "__main__":
    asyncio.run(test_search())