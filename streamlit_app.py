import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from llm import (
    load_tables_from_files,
    create_chunks,
    embed_and_index,
    retrieve_results,
    generate_llm_prompt,
    get_llm_answer
)

load_dotenv()

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

# Constants
FILES_TO_PROCESS = [
    "andhra_pradesh .json",
        "bihar .json",
        "Madhya_pradesh.json",
        "punjab .json",
        "all_india .json",
        "odisha.json",
        "Rajasthan.json",
        "Sikkim.json",
        "tamil_nadu.json",
        "telangana.json",
        "tripura.json",
        "Uttarakhand.json",
        "Uttar_Pradesh.json",
        "Arunachal_pradesh.json",
        "Assam.json",
        "Chhattisgarh.json",
        "Gujarat.json",
        "Harayan.json",
        "Himachal_pradesh.json"

]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
    
if 'rag_components' not in st.session_state:
    st.session_state.rag_components = None


@st.cache_resource(show_spinner="🚀 Initializing AI system...")
def initialize_rag_system():
    """Initialize the RAG system once and cache it"""
    try:
        tables = load_tables_from_files(FILES_TO_PROCESS)
        if not tables:
            raise ValueError("No valid tables loaded from files")
        
        chunks = create_chunks(tables)
        index, model, embeddings, chunks = embed_and_index(
            chunks,
            model_name='models/text-embedding-004',
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
        response = get_llm_answer(prompt)
        
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Main app
def main():
    # Initialize RAG system
    if not st.session_state.rag_initialized:
        rag_components = initialize_rag_system()
        if rag_components:
            st.session_state.rag_components = rag_components
            st.session_state.rag_initialized = True
        else:
            st.error("⚠️ Failed to initialize. Please check your data files.")
            st.stop()
    
    # Main chat area
    st.title("🤖 AI Chat Assistant")
    st.caption("Ask me anything about your data!")
    
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
                response = process_query(prompt)
            st.markdown(response)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()


if __name__ == "__main__":
    main()
