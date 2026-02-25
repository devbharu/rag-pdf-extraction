import sys
import os
import asyncio
from dotenv import load_dotenv

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), "Backend"))

try:
    from agents.orchestrator import Orchestrator
    from agents.master_agent import MasterAgent
    print("Imports successful.")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

async def test():
    try:
        print("Instantiating Orchestrator...")
        orch = Orchestrator()
        print("Orchestrator instantiated.")
        
        print("Instantiating MasterAgent...")
        master = MasterAgent()
        print("MasterAgent instantiated.")
        
        print("Verification successful (Instantiation only).")
    except Exception as e:
        print(f"Instantiation failed: {e}")

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not found in env, agents might fail if they try to connect.")
    
    asyncio.run(test())
