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
    layout="wide",
    initial_sidebar_state="expanded"
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
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1f1f1f;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    /* User messages - RIGHT aligned */
    div[data-testid="stChatMessageContent"]:has(+ div[data-testid="stChatMessage"]) {
        background-color: #007bff;
    }
    
    div.stChatMessage:has(div[data-testid="stChatMessageContent"]) {
        flex-direction: row-reverse;
        text-align: right;
    }
    
    div.stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        flex-direction: row-reverse;
    }
    
    /* History item styling */
    .history-item {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .history-item:hover {
        background-color: #f0f0f0;
        border-color: #007bff;
    }
    
    .history-timestamp {
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 0.25rem;
    }
    
    .history-preview {
        font-size: 0.9rem;
        color: #333;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
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

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = []

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = 0

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
            model_name='models/text-embedding-004',  # ✅ FIXED: Changed to Gemini embedding model
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


def save_current_session():
    """Save current chat session to history"""
    if len(st.session_state.messages) > 0:
        # Get first user message as preview
        first_message = next((msg['content'] for msg in st.session_state.messages if msg['role'] == 'user'), "New Chat")
        preview = first_message[:50] + "..." if len(first_message) > 50 else first_message
        
        session = {
            'id': st.session_state.current_session_id,
            'timestamp': datetime.now().strftime("%b %d, %I:%M %p"),
            'preview': preview,
            'messages': st.session_state.messages.copy()
        }
        
        # Check if session already exists
        existing = next((i for i, s in enumerate(st.session_state.chat_sessions) if s['id'] == session['id']), None)
        if existing is not None:
            st.session_state.chat_sessions[existing] = session
        else:
            st.session_state.chat_sessions.insert(0, session)


def load_session(session_id):
    """Load a chat session from history"""
    save_current_session()  # Save current before loading
    
    session = next((s for s in st.session_state.chat_sessions if s['id'] == session_id), None)
    if session:
        st.session_state.messages = session['messages'].copy()
        st.session_state.current_session_id = session_id


def start_new_chat():
    """Start a new chat session"""
    save_current_session()
    st.session_state.messages = []
    st.session_state.current_session_id = len(st.session_state.chat_sessions)


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
    
    # Sidebar - Chat History
    with st.sidebar:
        st.title("💬 Chat History")
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            start_new_chat()
            st.rerun()
        
        st.divider()
        
        # Display chat history
        if len(st.session_state.chat_sessions) == 0:
            st.info("No chat history yet. Start a conversation!")
        else:
            st.subheader("Previous Chats")
            for session in st.session_state.chat_sessions:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if st.button(
                        f"💬 {session['preview']}", 
                        key=f"session_{session['id']}",
                        use_container_width=True,
                        help=f"Created: {session['timestamp']}"
                    ):
                        load_session(session['id'])
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{session['id']}", help="Delete chat"):
                        st.session_state.chat_sessions = [s for s in st.session_state.chat_sessions if s['id'] != session['id']]
                        st.rerun()
                
                st.caption(f"🕒 {session['timestamp']}")
                st.markdown("---")
        
        # Stats at bottom
        st.divider()
        st.caption(f"📊 Total chats: {len(st.session_state.chat_sessions)}")
        st.caption(f"💬 Current messages: {len(st.session_state.messages)}")
    
    # Main chat area
    st.title("🤖 AI Chat Assistant")
    st.caption("Ask me anything about your data!")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
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
        
        # Auto-save session
        save_current_session()
        
        st.rerun()


if __name__ == "__main__":
    main()
