import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Import all necessary functions from the backend
from llm import (
    load_tables_from_files,
    create_chunks,
    embed_and_index,
    process_query_with_state_filter,
    get_json_files
)

load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="💬",
    layout="wide"
)

# Custom styling for chat alignment and UI
st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 900px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Chat input container */
    .stChatFloatingInputContainer {
        background-color: white;
        border-top: 1px solid #e0e0e0;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
    
if 'rag_components' not in st.session_state:
    st.session_state.rag_components = None

if 'model_answer' not in st.session_state:
    st.session_state.model_answer = None

@st.cache_resource(show_spinner="🚀 Initializing AI system...")
def initialize_rag_system():
    """Initialize the RAG system once and cache it"""
    try:
        # Get all JSON files from json_files directory
        json_dir = Path("json_files") / "json_files"
        
        if not json_dir.exists():
            # Try alternate path
            json_dir = Path("json_files")
        
        if not json_dir.exists():
            raise ValueError(f"Directory '{json_dir}' does not exist")
        
        # Get all JSON files (silently)
        json_files = get_json_files(str(json_dir))
        
        if not json_files:
            raise ValueError(f"No JSON files found in '{json_dir}'")
        
        # Load tables (silently)
        tables = load_tables_from_files(json_files)
        if not tables:
            raise ValueError("No valid tables loaded from files")
        
        # Create chunks (silently)
        chunks = create_chunks(tables)
        
        # Build index (silently)
        index, model = embed_and_index(
            chunks,
            model_name='models/text-embedding-004',
            batch_size=100
        )
        
        # Initialize Gemini model for answering
        model_answer = genai.GenerativeModel("gemini-2.5-flash")
        
        return {
            'index': index,
            'model': model,
            'chunks': chunks,
            'model_answer': model_answer,
            'tables': tables
        }
    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        return None

def process_query_wrapper(user_input):
    """Process user query using the upgraded backend"""
    try:
        rag = st.session_state.rag_components
        
        # Use the main process_query function from backend
        # This handles both simple and complex queries automatically
        result = process_query(
            user_input,
            rag['index'],
            rag['model'],
            rag['chunks'],
            rag['tables']
        )
        
        # Return just the final answer without metadata
        return result['final_answer']
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

# Main app
def main():
    # Initialize RAG system
    if not st.session_state.rag_initialized:
        rag_components = initialize_rag_system()
        if rag_components:
            st.session_state.rag_components = rag_components
            st.session_state.rag_initialized = True
            st.session_state.model_answer = rag_components['model_answer']
        else:
            st.error("⚠️ Failed to initialize. Please check your data files and API key.")
            st.stop()
    
    # Main chat area
    st.title("🤖 AI Chat Assistant")
    st.caption("Ask me anything about education statistics across Indian states!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here...", key="chat_input"):
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = process_query_wrapper(prompt)
            st.markdown(response)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

if __name__ == "__main__":
    main()
