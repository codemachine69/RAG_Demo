#!/usr/bin/env python3
"""
Simple RAG PDF Question Answering System

Usage: python rag_pdf.py document.pdf

Dependencies: pip install -U langchain langchain-community faiss-cpu pypdf sentence-transformers google-generativeai python-dotenv
"""

import os
import re
import sys
from typing import List, Dict, Any, Optional, Tuple
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.schema import Document
    import google.generativeai as genai
    import pypdf
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install -U langchain langchain-community faiss-cpu pypdf sentence-transformers google-generativeai python-dotenv")
    sys.exit(1)


def extract_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with page numbers."""
    text_content = []
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_content.append({
                    'page': page_num,
                    'text': text
                })
                
    logger.info(f"Extracted text from {len(text_content)} pages")
    return text_content


def create_chunks(text_content: List[Dict[str, Any]]) -> List[Document]:
    """Create text chunks from extracted PDF content."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    
    chunks = []
    for content in text_content:
        page = content['page']
        text = content['text']
        
        text_chunks = text_splitter.split_text(text)
        
        for chunk in text_chunks:
            chunks.append(Document(
                page_content=chunk,
                metadata={'page': page}
            ))
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def build_vector_index(chunks: List[Document]) -> FAISS:
    """Build FAISS vector index from chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    logger.info("Building vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("Vector index built")
    
    return vectorstore


def initialize_gemini() -> genai.GenerativeModel:
    """Initialize Gemini model."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment")
        print("Please set your API key in .env file")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Initialized Gemini 1.5 Flash")
    
    return model


def retrieve_context(vectorstore: FAISS, question: str, k: int = 4) -> Tuple[List[Document], str]:
    """Retrieve relevant context for a question."""
    results = vectorstore.similarity_search(question, k=k)
    
    # Format context with token management (max ~400 tokens)
    context_parts = []
    total_chars = 0
    max_chars = 1600  # ~400 tokens
    
    for doc in results:
        page = doc.metadata.get('page', 'Unknown')
        chunk_text = f"[Page {page}]\n{doc.page_content}"
        
        if total_chars + len(chunk_text) <= max_chars:
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        else:
            # Truncate to fit
            remaining = max_chars - total_chars
            if remaining > 100:
                truncated = doc.page_content[:remaining-50] + "..."
                context_parts.append(f"[Page {page}]\n{truncated}")
            break
    
    context = "\n\n".join(context_parts)
    logger.info(f"Retrieved {len(context_parts)} chunks (~{total_chars} chars)")
    
    return results, context


def answer_question(model: genai.GenerativeModel, question: str, context: str) -> str:
    """Generate answer using Gemini."""
    prompt = f"""Answer the following question based strictly on the provided context.

Context:
{context}

Question: {question}

Instructions: Answer from the provided context only. If the answer is not in the context, reply: "I don't know based on this document." Always include page citations.

Answer:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Error generating response."


def format_references(results: List[Document]) -> str:
    """Format page references."""
    pages = sorted(set(doc.metadata.get('page', 0) for doc in results))
    if pages:
        return f"Pages: {', '.join(map(str, pages))}"
    return "No references found"


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python rag_pdf.py document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found")
        sys.exit(1)
    
    try:
        # Initialize components
        print(f"Processing: {pdf_path}")
        
        # Extract and process PDF
        text_content = extract_pdf_text(pdf_path)
        chunks = create_chunks(text_content)
        vectorstore = build_vector_index(chunks)
        model = initialize_gemini()
        
        print("\n" + "="*60)
        print("RAG System Ready - Ask questions about the PDF")
        print("Type 'quit' to exit")
        print("="*60)
        
        # Interactive Q&A loop
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif not question:
                    continue
                
                print("Thinking...")
                
                # Get answer
                results, context = retrieve_context(vectorstore, question)
                answer = answer_question(model, question, context)
                references = format_references(results)
                
                print(f"\nAnswer: {answer}")
                print(f"{references}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"An error occurred: {e}")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()