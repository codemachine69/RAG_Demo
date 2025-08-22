# Simple RAG PDF Question Answering System

A lightweight, focused Retrieval-Augmented Generation (RAG) system for answering questions about PDF documents. **Ultra-simple CLI interface** with no unnecessary complexity.

## ✨ Features

- **📄 PDF Processing**: Extract text with page number preservation
- **✂️ Smart Chunking**: 1000 character chunks with 150 character overlap
- **🔍 Vector Search**: FAISS similarity search with sentence-transformers
- **🤖 AI Integration**: Google Gemini 1.5 Flash (fast & cost-effective)
- **📚 Grounded Answers**: Always based on document content with page citations
- **⚡ Single Mode**: One clean interactive Q&A interface

## 🚀 Installation

1. **Download** the `rag_pdf.py` script
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

**Command:**

```bash
python rag_pdf.py document.pdf
```

**Example:**
```bash
python rag_pdf.py M-618.pdf
```

## 🔑 Setup

1. **Get Gemini API Key**: https://makersuite.google.com/app/apikey
2. **Create `.env` file**:
   ```bash
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```
3. **Run the system** and start asking questions!

## 🎯 How It Works

1. **PDF Processing**: Extracts text from all pages
2. **Chunking**: Splits into manageable pieces with overlap
3. **Embeddings**: Creates semantic vectors using sentence-transformers
4. **Search**: Finds most relevant chunks for your question
5. **Answer**: Gemini generates grounded response with citations

<!-- ## 📊 Output Example

```
Question: What is USCIS?

Answer: USCIS's official website is www.uscis.gov (pages 11, 16). Their customer 
service number is 1-800-375-5283 or 1-800-767-1833 (for the hearing impaired) 
(page 11). To get forms, visit their website or call the USCIS Forms Line at 
1-800-870-3676 (page 11).

Pages: 11, 14, 16, 116
``` -->

## 🛠️ System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM
- **Storage**: ~100MB for FAISS index
- **Internet**: For model downloads (first run only)

## 🔧 Technical Details

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS (CPU optimized)
- **LLM**: Gemini 1.5 Flash (1M token context)
- **Chunk Size**: 1000 characters with 150 overlap
- **Context Limit**: ~400 tokens for optimal performance

## 📁 Project Structure

```
RAG_Demo/
├── rag_pdf.py          # Main application (181 lines)
├── requirements.txt     # Dependencies
├── .env                # Your API key (create this)
├── M-618.pdf          # Example document
└── faiss_index/       # Vector index (auto-created)
```

## 🚨 Troubleshooting

### **API Key Issues**
```bash
# Check if API key is loaded
echo $GOOGLE_API_KEY

# Or check .env file
cat .env
```

### **Dependencies**
```bash
# Reinstall if needed
pip install --force-reinstall -r requirements.txt
```

### **Memory Issues**
- Large PDFs may need more RAM
- Consider closing other applications

## 🎉 Why This Version is Better

| Before | After |
|--------|-------|
| 1,293 lines of code | **181 lines** |
| 6 CLI arguments | **1 argument** |
| Multiple modes | **Single mode** |
| Complex abstractions | **Simple functions** |
| Fallback logic | **Direct implementation** |
| Configuration options | **Smart defaults** |

**86% less code, 100% more focused!**

## 📝 License

This project is provided as-is for educational and research purposes.

---

**Simple. Fast. Effective.** 🎯
