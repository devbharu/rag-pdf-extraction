import asyncio
import os
from Backend.agents.sensor_agent import SensorAgent
from Backend.agents.history_agent import HistoryAgent

# Mock environment for testing if needed, but we are testing real DB
# os.environ["GROQ_API_KEY"] = "gsk_..." 

async def test_real_agents():
    print("--- Testing Real SensorAgent ---")
    sensor_agent = SensorAgent()
    # Test with a known equipment ID from the user's example
    equipment_id = "H_OP100" 
    print(f"Querying Sensor Data for: {equipment_id}")
    sensor_data = await sensor_agent.get_sensor_data(equipment_id)
    print(f"Result: {sensor_data}")

    print("\n--- Testing Real HistoryAgent ---")
    history_agent = HistoryAgent()
    # Test with the same equipment
    print(f"Querying History Data for: {equipment_id}")
    history_data = await history_agent.check_history([], equipment_id)
    print(f"Result: {history_data}")

if __name__ == "__main__":
    try:
        asyncio.run(test_real_agents())
    except Exception as e:
        print(f"Verification failed: {e}")
