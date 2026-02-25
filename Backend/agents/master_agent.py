import asyncio
import json
import os
from groq import Groq
from dotenv import load_dotenv

from .symptom_agent import SymptomAgent
from .sensor_agent import SensorAgent
from .retrieval_agent import RetrievalAgent
from .history_agent import HistoryAgent

load_dotenv()

class MasterAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "openai/gpt-oss-120b"
        
        self.symptom_agent = SymptomAgent()
        self.sensor_agent = SensorAgent()
        self.retrieval_agent = RetrievalAgent()
        self.history_agent = HistoryAgent()

    async def diagnose(self, query: str, user_id: int) -> dict:
        # Step 1: Run specialized agents in parallel
        # Note: Sensor and History agents might benefit from Symptom extraction, 
        # but for speed/parallelism we pass the raw query or handle simple extraction inside them.
        # For the mock, we'll do a quick extraction in the main flow or let them handle it.
        
        # We'll start Symptom extraction first as it's critical, or just run all.
        # To strictly follow "parallel", we launch all.
        
        # However, HistoryAgent needs symptoms to be accurate. 
        # But the prompt says "All 4 agents finish in parallel". 
        # This implies HistoryAgent might do its own extraction or fuzzy search.
        # For this implementation, we will run SymptomAgent first to get structured data, 
        # then run the others in parallel using that data, OR run all in parallel and combine.
        
        # Let's try running SymptomAgent first, then the others. It adds a small latency but ensures better quality.
        # User said "triggers 4 parallel agents simultaneously". 
        # I will launch them all. Sensor and History will do basic keyword matching on the query string for now.
        
        task_symptom = asyncio.create_task(self.symptom_agent.extract_symptoms(query))
        # For sensor/history, we pass the query. They will need to handle it.
        # My previous implementation of SensorAgent expects 'equipment' string.
        # I'll update the call to extract it from query or pass query.
        # Actually, let's just do a quick check here or modify SensorAgent to accept query.
        # I'll assume SensorAgent can handle the query string or I'll extract a keyword here.
        
        # Quick keyword extraction for the mock agents to allow parallelism
        equipment_keyword = "equipment"
        if "motor" in query.lower(): equipment_keyword = "motor"
        elif "pump" in query.lower(): equipment_keyword = "pump"
        
        task_sensor = asyncio.create_task(self.sensor_agent.get_sensor_data(equipment_keyword))
        task_retrieval = asyncio.create_task(self.retrieval_agent.retrieve_docs(query, user_id))
        
        # History agent needs symptoms.
        # The new signature is check_history(symptoms: list, equipment: str)
        task_history = asyncio.create_task(self.history_agent.check_history([query], equipment_keyword))

        # Wait for all
        symptoms, sensor_data, docs, history = await asyncio.gather(
            task_symptom, task_sensor, task_retrieval, task_history
        )

        # Step 2: Synthesize with Master Agent
        prompt = f"""
        You are the Master Diagnostic Agent for a manufacturing expert system.
        Synthesize the following evidence to provide a structured diagnosis.

        User Query: "{query}"

        Evidence:
        1. Symptoms (Extracted): {json.dumps(symptoms)}
        2. Sensor Data (Real-time): {json.dumps(sensor_data)}
        3. Relevant Documents (RAG): {json.dumps(docs)}
        4. Historical Issues: {json.dumps(history)}

        Reason step-by-step:
        - Analyze the symptoms and sensor data to identify anomalies.
        - Correlate with document knowledge and historical patterns.
        - Determine the most likely root causes.

        Output Format (JSON):
        {{
            "likely_causes": [
                {{"cause": "Name of cause", "probability": 0.0-1.0, "solution": "Recommended fix", "evidence": "Why you think this"}}
            ],
            "immediate_actions": ["Action 1", "Action 2"],
            "safety_warnings": ["Warning 1"],
            "confidence": 0.0-1.0,
            "document_refs": [{{"filename": "name", "page": "number"}}]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"MasterAgent error: {e}")
            # Fallback response
            return {
                "likely_causes": [],
                "immediate_actions": ["Contact supervisor"],
                "safety_warnings": ["Unknown error in diagnosis"],
                "confidence": 0.0,
                "error": str(e)
            }
