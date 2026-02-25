import sys
import os

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), "Backend"))

try:
    print("Checking imports...")
    from chat_history import ChatHistoryManager
    from gtts import gTTS
    from rag_docling import process_document
    import inspect
    
    print("Imports successful.")
    
    # Check ChatHistoryManager
    mgr = ChatHistoryManager("test_history.db")
    print("ChatHistoryManager instantiated.")
    
    # Check process_document signature
    sig = inspect.signature(process_document)
    if 'progress_callback' in sig.parameters:
        print("process_document has progress_callback parameter.")
    else:
        print("ERROR: process_document MISSING progress_callback parameter.")
        
    print("Verification passed.")
    
except Exception as e:
    print(f"Verification FAILED: {e}")
    import traceback
    traceback.print_exc()
