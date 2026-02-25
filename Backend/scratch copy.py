import streamlit as st
import os
import tempfile
from typing import List, TypedDict, Annotated
import operator

# --- LlamaIndex Imports ---
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# --- LangGraph & LangChain Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# --- Vector DB ---
import qdrant_client

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="Agentic Multimodal RAG", layout="wide")

st.title("🧠 Agentic RAG: LlamaIndex + LangGraph + Vision")
st.markdown("""
**Stack:** Groq (Llama 3.2 Vision & Llama 3.3 70B) | **FastEmbed (Local)** | Qdrant | LangGraph
""")

# Sidebar for API Keys
with st.sidebar:
    st.header("🔑 API Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    
    st.divider()
    st.info("Upload documents (PDF/Images) to start.")

if not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.stop()

# Set Environment Variables
os.environ["GROQ_API_KEY"] = groq_api_key

# ==========================================
# 2. CORE COMPONENTS INITIALIZATION
# ==========================================

@st.cache_resource
def get_llm():
    return Groq(model="llama-3.3-70b-versatile", temperature=0.1)

@st.cache_resource
def get_vision_model():
    return OpenAIMultiModal(
        model="llama-3.2-11b-vision-preview",
        api_key=groq_api_key,
        api_base="https://api.groq.com/openai/v1",
        max_new_tokens=1000,
    )

@st.cache_resource
def get_embed_model():
    # --- FIX: Removed st.toast from inside cached function ---
    return FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource
def initialize_vector_store():
    client = qdrant_client.QdrantClient(location=":memory:")
    return QdrantVectorStore(client=client, collection_name="agentic_rag_docs")

# Apply Global Settings
Settings.llm = get_llm()
Settings.embed_model = get_embed_model()

# ==========================================
# 3. MULTIMODAL INGESTION PIPELINE (OCR)
# ==========================================

def process_file_with_vision(uploaded_file):
    vision_model = get_vision_model()
    
    # Create temp file (Windows compatible)
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    file_ext = os.path.splitext(tmp_path)[1].lower()
    documents = []

    try:
        # Check if file is an image
        if file_ext in [".jpg", ".jpeg", ".png"]:
            st.toast(f"Processing Image: {uploaded_file.name}...", icon="👁️")
            
            # Load image using SimpleDirectoryReader
            img_reader = SimpleDirectoryReader(input_files=[tmp_path])
            img_docs = img_reader.load_data()
            
            # Prompt Llama 3.2 Vision
            prompt = (
                "Analyze this image. If it contains text, transcribe it exactly. "
                "If it contains charts or diagrams, describe the data trends and structure in detail. "
                "Output strictly in Markdown format."
            )
            
            response = vision_model.complete(prompt=prompt, image_documents=img_docs)
            
            doc = Document(text=str(response), metadata={"source": uploaded_file.name, "type": "image_ocr"})
            documents.append(doc)
            
        else:
            # Standard Text/PDF path
            st.toast(f"Processing Document: {uploaded_file.name}...", icon="📄")
            reader = SimpleDirectoryReader(input_files=[tmp_path])
            documents = reader.load_data()
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass 

    return documents

# ==========================================
# 4. INDEXING & RETRIEVAL ENGINE
# ==========================================

if "index" not in st.session_state:
    st.session_state.index = None

uploaded_files = st.sidebar.file_uploader("Upload Knowledge", accept_multiple_files=True, type=["pdf", "txt", "png", "jpg", "jpeg"])

if uploaded_files and st.sidebar.button("Process & Index"):
    all_docs = []
    vector_store = initialize_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    with st.spinner("Running Vision OCR and Indexing..."):
        for f in uploaded_files:
            docs = process_file_with_vision(f)
            all_docs.extend(docs)
        
        if all_docs:
            parser = MarkdownNodeParser()
            nodes = parser.get_nodes_from_documents(all_docs)
            
            # Create Index
            index = VectorStoreIndex(
                nodes, 
                storage_context=storage_context
            )
            st.session_state.index = index
            st.success(f"Indexed {len(all_docs)} documents into Qdrant!")
        else:
            st.warning("No documents were processed successfully.")

# ==========================================
# 5. LANGGRAPH AGENT DEFINITION
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str

def retrieve_knowledge(query: str) -> str:
    if st.session_state.index is None:
        return "No index found."
    
    retriever = st.session_state.index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query)
    return "\n\n".join([n.get_content() for n in nodes])

def retrieval_node(state: AgentState):
    messages = state["messages"]
    last_user_msg = messages[-1]
    
    if isinstance(last_user_msg, HumanMessage):
        query = last_user_msg.content
        context = retrieve_knowledge(query)
        return {"context": context, "messages": [SystemMessage(content=f"Retrieved Context:\n{context}")]}
    return {}

def generation_node(state: AgentState):
    messages = state["messages"]
    context = state.get("context", "")
    
    system_prompt = (
        "You are an intelligent assistant. Use the retrieved context to answer. "
        "If the context is irrelevant, state that you don't know."
    )
    
    prompt_str = f"{system_prompt}\n\nContext: {context}\n\n"
    for m in messages:
        if isinstance(m, HumanMessage):
            prompt_str += f"User: {m.content}\n"
        elif isinstance(m, AIMessage):
            prompt_str += f"Assistant: {m.content}\n"
    prompt_str += "Assistant:"
    
    llm = get_llm()
    response = llm.complete(prompt_str)
    return {"messages": [AIMessage(content=str(response))]}

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("generate", generation_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app_graph = workflow.compile()

# ==========================================
# 6. STREAMLIT CHAT UI
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.index is None:
        st.error("Please index documents first!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Agent is working..."):
                try:
                    inputs = {"messages": [HumanMessage(content=prompt)], "context": ""}
                    final_state = app_graph.invoke(inputs)
                    ai_msg = final_state["messages"][-1]
                    
                    st.markdown(ai_msg.content)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_msg.content})
                except Exception as e:
                    st.error(f"Error generating response: {e}")