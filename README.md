# DocuNexus (DocuSage)

**DocuNexus** is an advanced document analysis and question-answering platform for business documents. It supports OCR, visual document understanding, structured field extraction, and both text and visual search across uploaded documents. Built with Streamlit and state-of-the-art AI models (Donut, CLIP, LlamaIndex, Ollama), it enables users to upload, analyze, and query PDFs, images, and more.

## Features

- **Document Upload:** Supports PDF, TXT, DOCX, MD, JPG, PNG files.
- **OCR Extraction:** Extracts text from images and scanned PDFs using Tesseract and PyMuPDF.
- **Document Q&A:** Ask natural language questions about any uploaded document (text or image-based).
- **DocVQA (Donut):** Visual question answering for images and scanned documents using Donut (naver-clova-ix/donut-base-finetuned-docvqa).
- **Structured Field Extraction:** Extracts structured fields (e.g., invoice number, date, total) from images using Donut.
- **Visual Search:** Cross-document visual search using CLIP (openai/clip-vit-base-patch16). Find images by description.
- **Full-Text Search:** RAG-based search and retrieval using LlamaIndex.
- **Modern UI:** Streamlit interface for uploads, chat, field display, and search.

## Quickstart

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd DocuNexus
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```sh
   streamlit run DocuNexus.py
   ```

## Requirements
- Python 3.8+
- See `requirements.txt` for all Python dependencies
- Tesseract OCR (install system binary, e.g. `sudo apt install tesseract-ocr` or Windows installer)

## Usage
- Upload documents in the sidebar.
- Enable "Document QA" for RAG-based search.
- Use the chat to ask questions about your documents.
- For images, DocVQA and structured field extraction are automatic.
- Use "Index Visual Embeddings" and the search bar for cross-document visual search.

## Models Used
- **Donut (DocVQA):** `naver-clova-ix/donut-base-finetuned-docvqa`
- **CLIP (Visual Search):** `openai/clip-vit-base-patch16`
- **LLM (Text QA):** Ollama (Llama3, Mistral, Phi3, etc.)
- **Embeddings:** nomic-embed-text (via Ollama)

## Notes
- For best results, ensure Tesseract is installed and available in your PATH.
- Large models may require a GPU for reasonable performance.
- All processing is local; no data leaves your machine.

## License
MIT License

---

**DocuNexus** © 2025. Built with ❤️ for document intelligence.
