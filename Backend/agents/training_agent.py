import os
import json
from groq import Groq
from dotenv import load_dotenv
from .retrieval_agent import RetrievalAgent

load_dotenv()

class TrainingAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "openai/gpt-oss-120b"
        self.retrieval_agent = RetrievalAgent()

    async def generate_training(self, query: str, user_id: int) -> dict:
        """
        Generates a training module based on the query by retrieving relevant documents.
        """
        # 1. Retrieve relevant documents
        search_query = f"{query} procedure manual instructions training"
        docs = await self.retrieval_agent.retrieve_docs(search_query, user_id)
        
        # 2. Generate Training Content
        prompt = f"""
        You are a Technical Trainer. Create a training module based on the user's request and the provided context.
        
        User Request: "{query}"
        
        Context (from manuals):
        {json.dumps(docs)}
        
        Task:
        1. Define clear Learning Objectives.
        2. Create a Step-by-Step Guide/Procedure.
        3. Create a short Quiz (3 questions) to verify understanding.
        
        Output Format (JSON):
        {{
            "module_title": "Title of the training module",
            "learning_objectives": ["Obj 1", "Obj 2"],
            "steps": [
                {{"step": 1, "instruction": "Do this...", "warning": "Watch out for..."}}
            ],
            "quiz": [
                {{"question": "Q1?", "options": ["A", "B", "C"], "correct_answer": "A"}}
            ],
            "document_refs": [{{"filename": "name", "page": "number"}}]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "module_title": "Error",
                "message": f"Failed to generate training module: {str(e)}"
            }
