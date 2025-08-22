# RAG PDF Question Answering System

A complete Retrieval-Augmented Generation (RAG) system for answering questions about PDF documents. This system extracts text from PDFs, creates semantic embeddings, and uses AI to generate grounded answers based on the document content.

## Features

- **PDF Processing**: Extract text while preserving page numbers and detecting section headings
- **Smart Chunking**: Split text into ~1000 character chunks with 150 character overlap
- **Vector Search**: Use FAISS for efficient similarity search with sentence-transformers embeddings
- **Dual LLM Support**: 
  - Default: Hugging Face models (google/flan-t5-base)
  - Optional: Google Gemini API (if API key provided)
- **Grounding**: Forces answers to be based on retrieved context with citations
- **Multiple Modes**: Single question, demo, interactive Q&A, and batch processing

## Installation

1. **Clone or download** the `rag_pdf.py` script
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install -U langchain langchain-community faiss-cpu pypdf sentence-transformers transformers google-generativeai
   ```

## Usage

### Basic Usage

```bash
# Interactive mode (default)
python rag_pdf.py --pdf your_document.pdf

# Ask a single question
python rag_pdf.py --pdf your_document.pdf --ask "What is the main topic?"

# Run demo with sample questions
python rag_pdf.py --pdf your_document.pdf --demo

# Rebuild index from scratch
python rag_pdf.py --pdf your_document.pdf --rebuild

# Custom index directory
python rag_pdf.py --pdf your_document.pdf --persist_dir ./my_index
```

### Environment Variables

For Gemini API support, set your Google API key:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### Examples

#### Example 1: Interactive Q&A
```bash
python rag_pdf.py --pdf M-618.pdf
```
This starts an interactive session where you can ask questions about the PDF.

#### Example 2: Single Question
```bash
python rag_pdf.py --pdf M-618.pdf --ask "What are the key requirements?"
```

#### Example 3: Demo Mode
```bash
python rag_pdf.py --pdf M-618.pdf --demo
```
Runs through a set of sample questions to demonstrate the system.

## How It Works

1. **PDF Ingestion**: 
   - Extracts text from each page
   - Detects section headings (ALL-CAPS patterns)
   - Preserves page numbers and metadata

2. **Text Chunking**:
   - Splits text into ~1000 character chunks
   - Maintains 150 character overlap between chunks
   - Preserves page and section metadata

3. **Vector Indexing**:
   - Generates embeddings using sentence-transformers/all-MiniLM-L6-v2
   - Stores vectors in FAISS index
   - Persists index to disk for reuse

4. **Retrieval**:
   - Searches for top-k relevant chunks (default k=4)
   - Uses semantic similarity to find context

5. **Answer Generation**:
   - Sends question + retrieved context to LLM
   - Forces grounding with specific prompt instructions
   - Includes page and section citations

## Output Format

Each answer includes:
- **Answer**: The LLM-generated response
- **References**: List of sections and pages used as context
- **Context Chunks**: Option to view the actual retrieved text chunks

## System Requirements

- **Python**: 3.8+
- **Memory**: At least 4GB RAM (more for large PDFs)
- **Storage**: Space for FAISS index (typically 100MB-1GB depending on PDF size)
- **Internet**: Required for first-time model downloads

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed correctly
2. **Memory Issues**: Large PDFs may require more RAM
3. **Model Download**: First run will download embedding models (~100MB)
4. **FAISS Issues**: Try reinstalling with `pip install --force-reinstall faiss-cpu`

### Performance Tips

- Use `--rebuild` only when necessary
- Index is automatically saved and reused
- Consider using smaller chunk sizes for very large documents
- Gemini API provides faster responses than local models

## Architecture

The system is built with a modular architecture:

- **PDFProcessor**: Handles PDF text extraction and section detection
- **TextChunker**: Manages text chunking with metadata preservation
- **VectorIndex**: Manages embeddings and FAISS operations
- **LLMBackend**: Abstract interface for different LLM providers
- **RAGSystem**: Main orchestrator that coordinates all components

## Customization

### Changing Chunk Size
Modify the `TextChunker` class parameters:
```python
chunker = TextChunker(chunk_size=800, chunk_overlap=100)
```

### Using Different Models
Change the embedding model in `VectorIndex`:
```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

### Custom Prompts
Modify the prompt templates in the LLM classes for different response styles.

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.
