from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

class Orchestrator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "openai/gpt-oss-120b" # Using the same model as main.py

    async def detect_intent(self, query: str) -> str:
        prompt = f"""
        You are an intelligent orchestrator for a manufacturing support system.
        Analyze the user's input and classify the intent into one of the following categories:
        
        1. "diagnostic": The user is reporting a problem, failure, or symptom with equipment (e.g., "Motor won't start", "Strange noise", "Vibration high").
        2. "compliance": The user is asking about safety rules, regulations, or standard operating procedures (e.g., "What is the safety protocol for...", "ISO standards for...").
        3. "training": The user is asking for educational content or how-to guides (e.g., "How do I calibrate...", "Explain the working principle of...").
        4. "other": Any other query that doesn't fit the above.

        User Input: "{query}"

        Return ONLY the category name (diagnostic, compliance, training, or other) as a lowercase string. Do not add any punctuation or explanation.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Orchestrator error: {e}")
            return "other"
