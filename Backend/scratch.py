import streamlit as st
import os
import tempfile
import base64
from typing import List, TypedDict, Annotated
import operator
import json


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


# --- LangGraph & LangChain Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


# --- Vector DB ---
import qdrant_client
from groq import Groq as GroqClient


# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="Agentic Multimodal RAG", layout="wide")

st.title("🧠 Agentic RAG: LlamaIndex + LangGraph + Vision")
st.markdown("""
**Stack:** Groq (Llama 4 Scout/Maverick Vision) | **FastEmbed (Local)** | Qdrant | LangGraph
""")


# Sidebar for API Keys
with st.sidebar:
    st.header("🔑 API Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    
    st.divider()
    st.info("ℹ️ **Available Vision Models:**\n\n"
            "- `meta-llama/llama-4-scout-17b-16e-instruct` (Fast, 594 TPS)\n"
            "- `meta-llama/llama-4-maverick-17b-128e-instruct` (Powerful, 562 TPS)\n"
            "- `llava-v1.5-7b-4096-preview` (Lightweight)")


if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")
    st.stop()


# Set Environment Variables
os.environ["GROQ_API_KEY"] = groq_api_key


# ==========================================
# 2. CORE COMPONENTS INITIALIZATION
# ==========================================


@st.cache_resource
def get_llm():
    """Get Groq LLM for text generation"""
    return Groq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=groq_api_key)


@st.cache_resource
def get_groq_client():
    """Get native Groq client for vision tasks"""
    return GroqClient(api_key=groq_api_key)


@st.cache_resource
def get_embed_model():
    """Get embedding model for document indexing"""
    return FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource
def initialize_vector_store():
    """Initialize in-memory Qdrant vector store"""
    client = qdrant_client.QdrantClient(location=":memory:")
    return QdrantVectorStore(client=client, collection_name="agentic_rag_docs")


# Apply Global Settings
Settings.llm = get_llm()
Settings.embed_model = get_embed_model()


# ==========================================
# 3. MULTIMODAL INGESTION PIPELINE (OCR)
# ==========================================


def process_image_with_vision(image_path: str, filename: str) -> str:
    """Process image using Groq Llama 4 Scout vision model"""
    groq_client = get_groq_client()
    
    try:
        # Read image and encode to base64
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        # Determine mime type
        ext = os.path.splitext(filename)[1].lower()
        mime_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        mime_type = mime_type_map.get(ext, "image/jpeg")
        
        st.toast(f"📸 Processing Image: {filename} with Llama 4 Scout...", icon="👁️")
        
        # Call Groq vision API
        message = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Fast & accurate
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image in detail. If it contains text, transcribe it exactly. "
                                "If it contains charts, diagrams, or data visualizations, describe the structure and trends. "
                                "If it contains tables, extract and format as markdown. "
                                "Output strictly in Markdown format."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2048,
            temperature=0.2
        )
        
        extracted_text = message.choices[0].message.content
        st.toast(f"✅ Successfully processed {filename}", icon="🎉")
        
        return extracted_text
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return ""


def process_file_with_vision(uploaded_file):
    """Process uploaded files (images or documents)"""
    
    # Create temp file (Windows compatible)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    file_ext = os.path.splitext(tmp_path)[1].lower()
    documents = []

    try:
        # Handle images with vision model
        if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            extracted_text = process_image_with_vision(tmp_path, uploaded_file.name)
            
            if extracted_text:
                doc = Document(
                    text=extracted_text,
                    metadata={
                        "source": uploaded_file.name,
                        "type": "image_ocr",
                        "model": "llama-4-scout-vision"
                    }
                )
                documents.append(doc)
        
        # Handle text/PDF files
        elif file_ext in [".pdf", ".txt", ".md"]:
            st.toast(f"📄 Processing Document: {uploaded_file.name}...", icon="📄")
            reader = SimpleDirectoryReader(input_files=[tmp_path])
            documents = reader.load_data()
            st.toast(f"✅ Successfully processed {uploaded_file.name}", icon="🎉")
        
        else:
            st.warning(f"⚠️ Unsupported file type: {file_ext}")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
    
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


uploaded_files = st.sidebar.file_uploader(
    "📤 Upload Knowledge Base",
    accept_multiple_files=True,
    type=["pdf", "txt", "md", "png", "jpg", "jpeg", "gif", "webp"]
)


if uploaded_files and st.sidebar.button("🚀 Process & Index", use_container_width=True):
    all_docs = []
    vector_store = initialize_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    with st.spinner("🔄 Running Vision OCR and Indexing..."):
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
            st.success(f"✅ Indexed {len(all_docs)} documents into Qdrant!")
        else:
            st.warning("⚠️ No documents were processed successfully.")


# ==========================================
# 5. LANGGRAPH AGENT DEFINITION
# ==========================================


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str


def retrieve_knowledge(query: str) -> str:
    """Retrieve relevant documents from vector store"""
    if st.session_state.index is None:
        return "❌ No index found. Please upload and index documents first."
    
    retriever = st.session_state.index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query)
    
    if not nodes:
        return "❌ No relevant documents found."
    
    return "\n\n".join([n.get_content() for n in nodes])


def retrieval_node(state: AgentState):
    """Retrieval step of the agent"""
    messages = state["messages"]
    last_user_msg = messages[-1]
    
    if isinstance(last_user_msg, HumanMessage):
        query = last_user_msg.content
        context = retrieve_knowledge(query)
        return {"context": context, "messages": [SystemMessage(content=f"Retrieved Context:\n{context}")]}
    return {}


def generation_node(state: AgentState):
    """Generation step of the agent"""
    messages = state["messages"]
    context = state.get("context", "")
    
    system_prompt = (
        "You are an intelligent assistant with access to a knowledge base. "
        "Use the retrieved context to answer questions accurately. "
        "If the context doesn't contain relevant information, state that clearly. "
        "Always cite the source when using information from documents."
    )
    
    prompt_str = f"{system_prompt}\n\nContext:\n{context}\n\n"
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


st.subheader("💬 Chat with Your Documents")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask about your documents or images..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.index is None:
        st.error("❌ Please upload and index documents first!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("⚙️ Agent is thinking..."):
                try:
                    inputs = {"messages": [HumanMessage(content=prompt)], "context": ""}
                    final_state = app_graph.invoke(inputs)
                    ai_msg = final_state["messages"][-1]
                    
                    response_text = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
                    st.markdown(response_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    st.info(f"**Debug Info:** {type(e).__name__}: {str(e)}")


# Footer with model info
st.divider()
st.caption("🤖 Powered by Groq Llama 4 Scout Vision | 📚 Vector DB: Qdrant | 🧠 Orchestration: LangGraph")