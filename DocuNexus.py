import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import logging
import time
import os
import tempfile  
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for PDF image extraction
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None
if 'visual_index' not in st.session_state:
    st.session_state.visual_index = []  # List of dicts: {filename, embedding, file}

# Initialize LLM and embedding model
def initialize_models(model_name):
    Settings.llm = Ollama(model=model_name, request_timeout=300.0, max_tokens=512)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.chunk_size = 1024

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"OCR error: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            # Try text extraction first
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
            else:
                # If no text, do OCR on page image
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logging.error(f"PDF OCR error: {str(e)}")
        return ""

# Function to handle document upload and create vector store


def handle_document_upload(uploaded_files):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_texts = []
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                ext = os.path.splitext(file.name)[1].lower()
                if ext in [".jpg", ".jpeg", ".png"]:
                    # OCR for images
                    text = extract_text_from_image(file_path)
                    doc_texts.append({"filename": file.name, "text": text})
                elif ext == ".pdf":
                    # Try text extraction, fallback to OCR
                    text = extract_text_from_pdf(file_path)
                    doc_texts.append({"filename": file.name, "text": text})
                else:
                    # Text-based files
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as tf:
                        text = tf.read()
                    doc_texts.append({"filename": file.name, "text": text})
            # Save all extracted texts as .txt for indexing
            for doc in doc_texts:
                txt_path = os.path.join(temp_dir, doc["filename"] + ".txt")
                with open(txt_path, "w", encoding="utf-8") as out_f:
                    out_f.write(doc["text"])
            documents = SimpleDirectoryReader(temp_dir).load_data()
            index = VectorStoreIndex.from_documents(documents)
        return index
    except Exception as e:
        logging.error(f"Document processing error: {str(e)}")
        raise e

# Load Donut model and processor (global cache for performance)
@st.cache_resource(show_spinner=False)
def get_donut_model():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model.eval()
    return processor, model

def run_docvqa_on_image(image_file, question):
    try:
        processor, model = get_donut_model()
        image = Image.open(image_file).convert("RGB")
        task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        pixel_values = processor(image, return_tensors="pt").pixel_values
        input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=input_ids,
                max_length=128,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Clean up answer
        answer = answer.replace(task_prompt, "").strip()
        return answer
    except Exception as e:
        logging.error(f"DocVQA error: {str(e)}")
        return "[DocVQA error: could not extract answer]"

# Load CLIP model for visual search (global cache)
@st.cache_resource(show_spinner=False)
def get_clip_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    model.eval()
    return processor, model

def extract_structured_fields_from_image(image_file, task="invoice"):
    """Extract structured fields from an image using Donut (e.g., invoice fields)."""
    try:
        processor, model = get_donut_model()
        image = Image.open(image_file).convert("RGB")
        if task == "invoice":
            prompt = "<s_invoice>"
        else:
            prompt = "<s_receipt>"
        pixel_values = processor(image, return_tensors="pt").pixel_values
        input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=input_ids,
                max_length=512,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Try to parse as JSON if possible
        import json
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end != -1:
                fields = json.loads(result[start:end])
            else:
                fields = {"raw": result}
        except Exception:
            fields = {"raw": result}
        return fields
    except Exception as e:
        logging.error(f"Field extraction error: {str(e)}")
        return {"error": str(e)}

def index_visual_embeddings(uploaded_files):
    processor, model = get_clip_model()
    visual_index = []
    for file in uploaded_files:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            image = Image.open(file).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                emb = model.get_image_features(**inputs).cpu().numpy()[0]
            visual_index.append({"filename": file.name, "embedding": emb, "file": file})
    st.session_state.visual_index = visual_index

def visual_search(query, top_k=3):
    processor, model = get_clip_model()
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_emb = model.get_text_features(**inputs).cpu().numpy()[0]
    # Compute cosine similarity
    import numpy as np
    results = []
    for doc in st.session_state.visual_index:
        sim = np.dot(query_emb, doc["embedding"]) / (np.linalg.norm(query_emb) * np.linalg.norm(doc["embedding"]))
        results.append((sim, doc))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]

# Modified stream_chat function with RAG
def stream_chat(model, messages, use_rag=False, docvqa_images=None, docvqa_question=None):
    try:
        # If docvqa_images and question provided, run DocVQA
        if docvqa_images and docvqa_question:
            answers = []
            for img_file in docvqa_images:
                answer = run_docvqa_on_image(img_file, docvqa_question)
                answers.append(f"**{os.path.basename(getattr(img_file, 'name', str(img_file)))}**: {answer}")
            return "\n".join(answers)
        llm = Ollama(model=model, request_timeout=300.0, max_tokens=512)
        response = ""
        if use_rag and st.session_state.vector_index:
            last_user_msg = next(msg for msg in reversed(messages) if msg.role == "user")
            query_engine = st.session_state.vector_index.as_query_engine()
            context = query_engine.query(last_user_msg.content)
            rag_prompt = f"Context: {context}\n\nQuestion: {last_user_msg.content}"
            messages[-1].content = rag_prompt
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
            "Upload documents for QA (PDF, TXT, DOCX, MD, JPG, PNG)", 
            type=["pdf", "txt", "docx", "md", "jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        use_rag = st.toggle("Enable Document QA", value=False)
        if uploaded_files:
            if st.button("Index Visual Embeddings"):
                with st.spinner("Indexing images for visual search..."):
                    index_visual_embeddings(uploaded_files)
                st.success("Visual embeddings indexed!")

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

    # Visual search bar
    if st.session_state.visual_index:
        st.subheader("Cross-document Visual Search")
        visual_query = st.text_input("Search images by description (e.g. 'invoice with total $1000'):")
        if visual_query:
            results = visual_search(visual_query)
            st.write("Top matches:")
            for sim, doc in results:
                st.image(doc["file"], caption=f"{doc['filename']} (score: {sim:.2f})", width=300)

    # Chat interface
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Detect if any uploaded file is an image and the question is likely about an image doc
        image_files = [f for f in (uploaded_files or []) if os.path.splitext(f.name)[1].lower() in [".jpg", ".jpeg", ".png"]]
        # If there are image files and RAG is enabled, offer DocVQA and field extraction
        if image_files and use_rag:
            with st.chat_message("assistant"):
                start_time = time.time()
                with st.spinner("Running DocVQA and extracting fields from images..."):
                    try:
                        response = stream_chat(
                            model,
                            [],  # not using LLM chat history for DocVQA
                            use_rag=False,
                            docvqa_images=image_files,
                            docvqa_question=prompt
                        )
                        # Structured field extraction for each image
                        st.write(response)
                        for img_file in image_files:
                            st.markdown(f"**Structured Fields for {img_file.name}:**")
                            fields = extract_structured_fields_from_image(img_file, task="invoice")
                            st.json(fields)
                        duration = time.time() - start_time
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"[DocVQA and field extraction complete]\n\n_Duration: {duration:.2f}s_"
                        })
                    except Exception as e:
                        st.error(f"DocVQA/Field extraction error: {str(e)}")
                        logging.error(f"DocVQA/Field extraction error: {str(e)}")
        else:
            # Generate response as usual
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