import sys
import os

# Ensure Backend directory is in sys.path to allow imports from sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from rag_docling import search_chunks

class RetrievalAgent:
    def __init__(self):
        pass

    async def retrieve_docs(self, query: str, user_id: int) -> list:
        """
        Retrieves relevant document chunks using the existing RAG pipeline.
        """
        try:
            # Search for relevant chunks
            # We don't filter by doc_id here to search all user's documents
            chunks = search_chunks(
                query=query,
                user_id=user_id,
                doc_id=None,
                k=3  # Top 3 most relevant chunks
            )
            
            # Format the results for the Master Agent
            results = []
            for chunk in chunks:
                results.append({
                    "filename": chunk.get("metadata", {}).get("filename", "Unknown"),
                    "page": chunk.get("page", "Unknown"),
                    "content": chunk.get("text", "")[:500] + "...", # Truncate for brevity
                    "score": chunk.get("_distance", 0) # LanceDB returns distance, lower is better usually, or score
                })
            return results
        except Exception as e:
            print(f"RetrievalAgent error: {e}")
            return []
