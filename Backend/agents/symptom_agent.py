from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

class SymptomAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "openai/gpt-oss-120b"

    async def extract_symptoms(self, query: str) -> dict:
        prompt = f"""
        You are a specialized Symptom Extraction Agent.
        Analyze the user's input and extract structured data about the equipment failure.
        
        User Input: "{query}"

        Extract the following fields:
        - symptom: The main issue (e.g., "humming", "no_rotation", "overheating"). Combine multiple if present.
        - equipment: The specific equipment mentioned (e.g., "motor", "pump", "conveyor").
        - severity: Estimate severity (low, medium, high, critical) based on the description.
        - duration: Any duration mentioned (e.g., "5min", "2 days"). If not mentioned, use null.
        - status: Current state (e.g., "running", "stopped", "jammed").

        Return the result as a valid JSON object. Do not include markdown formatting like ```json.
        Example:
        {{
            "symptom": "humming+no_rotation",
            "equipment": "motor",
            "severity": "high",
            "duration": "5min",
            "status": "stopped"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            # Clean up potential markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content)
        except Exception as e:
            print(f"SymptomAgent error: {e}")
            return {"error": str(e)}
