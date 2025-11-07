import streamlit as st
import os
from llm import (
    load_tables_from_files,
    create_chunks,
    embed_and_index,
    retrieve_results,
    generate_llm_prompt,
    get_llm_answer
)

# Configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Minimal custom styling (only essentials)
st.markdown("""
<style>
    /* Remove extra padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Chat input styling */
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
FILES_TO_PROCESS = [
    "punjab.json",
    "all_india.json",
    "andhra_pradesh.json",
    "bihar.json",
    "UP.json",
    "MP.json"
]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
    
if 'rag_components' not in st.session_state:
    st.session_state.rag_components = None

if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "AIzaSyA7oR7AJEDwQYDMMQGjHc1oLve_BUkC-h4")


@st.cache_resource(show_spinner="Initializing AI system...")
def initialize_rag_system():
    """Initialize the RAG system once and cache it"""
    try:
        tables = load_tables_from_files(FILES_TO_PROCESS)
        if not tables:
            raise ValueError("No valid tables loaded from files")
        
        chunks = create_chunks(tables)
        index, model, embeddings, chunks = embed_and_index(
            chunks,
            model_name='all-MiniLM-L6-v2',
            file_paths=FILES_TO_PROCESS,
            use_cache=True
        )
        
        return {
            'index': index,
            'model': model,
            'embeddings': embeddings,
            'chunks': chunks
        }
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None


def process_query(user_input):
    """Process user query and return AI response"""
    try:
        rag = st.session_state.rag_components
        
        # Retrieve relevant context
        retrieved = retrieve_results(
            user_input, 
            rag['index'], 
            rag['model'], 
            rag['chunks'], 
            top_k=3
        )
        
        # Generate prompt and get response
        prompt = generate_llm_prompt(retrieved, user_input)
        response = get_llm_answer(prompt, api_key=st.session_state.api_key)
        
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Main app
def main():
    # Title
    st.title("💬 AI Chat Assistant")
    
    # Initialize RAG system
    if not st.session_state.rag_initialized:
        rag_components = initialize_rag_system()
        if rag_components:
            st.session_state.rag_components = rag_components
            st.session_state.rag_initialized = True
        else:
            st.error("⚠️ Failed to initialize. Please check your data files.")
            st.stop()
    
    # Sidebar with controls
    with st.sidebar:
        st.header("Chat Controls")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Chat history count
        msg_count = len(st.session_state.messages)
        st.caption(f"📊 Messages: {msg_count}")
        
        if msg_count == 0:
            st.info("Start a conversation to see history")
    
    # Display chat history using native components
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input (native Streamlit component)
    if prompt := st.chat_input("Ask me anything..."):
        # Validate API key
        if not st.session_state.api_key:
            st.error("⚠️ API key not found. Set GEMINI_API_KEY environment variable.")
            st.stop()
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_query(prompt)
            st.markdown(response)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
