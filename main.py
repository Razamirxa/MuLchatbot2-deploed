# from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from more_itertools import chunked

# Page configuration
st.set_page_config(
    page_title="MUL University Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (from original code)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stSidebar {
        background-color: #1e1e1e !important;
        color: #ffffff;
    }
    .stChatMessage {
        background-color: ##2d2d2d !important;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .status-success {
        color: #4caf50;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ff9800;
        font-weight: bold;
    }
    
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
    
    /* Additional styling for better appearance */
    .stChatMessage {
        background-color: transparent !important;
    }
    

    
    /* User message styling */
    .stChatMessage[data-testid="user-message"] > div {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant-message"] > div {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Header (from original code)
st.markdown("""
<div class="main-header">
    <h1>🎓 Minhaj University Assistant</h1>
    <p>Your AI-powered guide to MUL University information</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for system status and additional features (from original code)
with st.sidebar:
    # Logo and title
    
    st.image("https://www.mul.edu.pk/images/logo-mul-footer.png")
    
    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    st.markdown("""
    - What programs does MUL offer?
    - Tell me about admission requirements
    - What is the fee structure (your program name)?
    """)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.langchain_messages = []
        st.rerun()

# Import your chain
from chain import chain

prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Main Content

if len(msgs.messages) == 0:
    msgs.add_ai_message("Assalam o Alaikum! How can I assist you today?")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("Ask me anything about MUL University..."):
    st.chat_message("human").write(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            _chat_history = st.session_state.langchain_messages[1:40]
            _chat_history_tranform = list(
                chunked([msg.content for msg in _chat_history], n=2)
            )
            
            response = chain.stream(
                {"question": prompt, "chat_history": _chat_history_tranform}
            )
            
            for res in response:
                full_response += res or ""
                message_placeholder.markdown(full_response + "▋")  # Changed cursor to match original
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
            
            msgs.add_user_message(prompt)
            msgs.add_ai_message(full_response)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
