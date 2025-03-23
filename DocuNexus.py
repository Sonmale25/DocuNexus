import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import logging
import time
import os
import tempfile  

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None

# Initialize LLM and embedding model
def initialize_models(model_name):
    Settings.llm = Ollama(model=model_name, request_timeout=300.0, max_tokens=512)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.chunk_size = 1024

# Function to handle document upload and create vector store


def handle_document_upload(uploaded_files):
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            # Load and index documents
            documents = SimpleDirectoryReader(temp_dir).load_data()
            index = VectorStoreIndex.from_documents(documents)
        
        return index
    except Exception as e:
        logging.error(f"Document processing error: {str(e)}")
        raise e

# Modified stream_chat function with RAG
def stream_chat(model, messages, use_rag=False):
    try:
        llm = Ollama(model=model, request_timeout=300.0, max_tokens=512)
        response = ""
        
        if use_rag and st.session_state.vector_index:
            # RAG workflow
            last_user_msg = next(msg for msg in reversed(messages) if msg.role == "user")
            query_engine = st.session_state.vector_index.as_query_engine()
            context = query_engine.query(last_user_msg.content)
            
            # Augment prompt with context
            rag_prompt = f"Context: {context}\n\nQuestion: {last_user_msg.content}"
            messages[-1].content = rag_prompt
            
        # Stream response
        resp = llm.stream_chat(messages)
        response_placeholder = st.empty()
        
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
            
        return response
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise e

def main():
    st.title("Document-QA Chat with LLMs")
    logging.info("App started")

    # Sidebar controls
    with st.sidebar:
        model = st.selectbox("Choose model", ["llama3:latest", "mistral:latest", "phi3"])
        uploaded_files = st.file_uploader(
            "Upload documents for QA", 
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True
        )
        use_rag = st.toggle("Enable Document QA", value=False)

    # Initialize models
    initialize_models(model)

    # Handle document upload
    if uploaded_files and use_rag:
        with st.spinner("Processing documents..."):
            try:
                st.session_state.vector_index = handle_document_upload(uploaded_files)
                st.success(f"Loaded {len(uploaded_files)} documents!")
            except Exception as e:
                st.error(f"Document error: {str(e)}")

    # Chat interface
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Generate response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                
                with st.spinner("Analyzing..."):
                    try:
                        messages = [
                            ChatMessage(role=msg["role"], content=msg["content"])
                            for msg in st.session_state.messages
                        ]
                        
                        response = stream_chat(
                            model, 
                            messages, 
                            use_rag=use_rag and st.session_state.vector_index
                        )
                        
                        duration = time.time() - start_time
                        full_response = f"{response}\n\n_Duration: {duration:.2f}s_"
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logging.error(f"Generation error: {str(e)}")

if __name__ == "__main__":
    main()