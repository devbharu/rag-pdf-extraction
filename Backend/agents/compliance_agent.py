import os
import json
from groq import Groq
from dotenv import load_dotenv
from .retrieval_agent import RetrievalAgent

load_dotenv()

class ComplianceAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "openai/gpt-oss-120b" # Or a suitable model
        self.retrieval_agent = RetrievalAgent()

    async def check_compliance(self, query: str, user_id: int) -> dict:
        """
        Checks compliance based on the query by retrieving relevant documents
        and verifying against standards.
        """
        # 1. Retrieve relevant documents
        # We append "safety standards compliance regulations" to the query to improve retrieval relevance
        search_query = f"{query} safety standards compliance regulations"
        docs = await self.retrieval_agent.retrieve_docs(search_query, user_id)
        
        # 2. Synthesize Compliance Report
        prompt = f"""
        You are a Compliance & Safety Officer. Analyze the following context to answer the user's query regarding compliance.
        
        User Query: "{query}"
        
        Context (from manuals/standards):
        {json.dumps(docs)}
        
        Task:
        1. Identify applicable safety standards (ISO, OSHA, internal).
        2. List mandatory Personal Protective Equipment (PPE).
        3. Highlight potential risks and non-compliance penalties if mentioned.
        4. Provide a clear "Compliant" or "Non-Compliant" assessment if possible, or "Requires Verification".
        
        Output Format (JSON):
        {{
            "assessment": "Compliant/Non-Compliant/Requires Verification",
            "standards": ["Standard 1", "Standard 2"],
            "required_ppe": ["PPE 1", "PPE 2"],
            "risks": ["Risk 1", "Risk 2"],
            "recommendations": ["Rec 1", "Rec 2"],
            "document_refs": [{{"filename": "name", "page": "number"}}]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Low temperature for strict adherence to facts
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "assessment": "Error",
                "message": f"Failed to generate compliance report: {str(e)}"
            }
