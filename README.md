# Chatbot with RAG

> **AI-powered document assistant** that lets you upload PDFs and ask questions about their content using Retrieval-Augmented Generation (RAG) with Streamlit, LangChain, and Google Gemini.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4)](https://deepmind.google/technologies/gemini/)
[![LangChain](https://img.shields.io/badge/LangChain-Community-1f425f)](https://python.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-Vector_DB-FF8A65)](https://www.trychroma.com/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🎯 Overview

**AI PDF Chatbot with RAG** is a question-answering system that lets users upload PDF documents and converse with them. It uses LangChain to extract and process text, Chroma for vector storage, and Google Gemini for creating embeddings and generating intelligent answers based on the document's content, all wrapped in an interactive Streamlit frontend.

### Key Capabilities

- **Vector Search**: Split and store document content in a Chroma database for semantic search
- **Google Gemini Integration**: Utilizes Gemini for accurate embeddings and natural language generation
- **RAG Pipeline**: Implements Retrieval-Augmented Generation to provide context-aware answers

---

## ✨ Features

- ✅ **Document Upload** - Upload PDFs to build a custom knowledge base
- ✅ **Intelligent Chat** - Ask questions about your documents
- ✅ **Contextual Answers** - AI responses are grounded in the document content
- ✅ **Google Gemini Power** - Uses `gemini-2.5-flash` for fast and accurate responses
- ✅ **LangChain Framework** - Robust text splitting and vector database integration

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│ Web Frontend │
│ (Streamlit) │
└───────────────────────┬─────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│ PDF Processing & Chunking │
│ • PyPDF2 for text extraction │
│ • LangChain RecursiveCharacterTextSplitter │
└───────────────────────┬─────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│ Vector Database (Chroma) │
│ • Embeddings via Google Generative AI │
│ • Stores document chunks for similarity search │
└───────────────────────┬─────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│ LangChain RAG Pipeline │
│ • Retrieves relevant context │
│ • Gemini 2.5 Flash for answer generation │
└─────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Upload**: User uploads a PDF via the Streamlit interface
2. **Text Processing**: Application reads the PDF and splits it into manageable chunks
3. **Embedding**: Text chunks are converted to embeddings using Google Gemini Embeddings
4. **Storage**: Embeddings are stored in an in-memory Chroma vector database
5. **User Query**: User asks a question about the document
6. **Retrieval**: Relevant chunks are retrieved from Chroma based on similarity to the query
7. **Generation**: Gemini 2.5 Flash synthesizes the retrieved chunks and the query to generate a helpful answer
8. **Display**: The response is presented to the user in the chat interface

---

## 📦 Setup & Usage

**Prerequisites**: Python 3.10+ and a [Google Gemini API Key](https://aistudio.google.com/).

```bash
git clone https://github.com/yourusername/chatbot-with-RAG.git
cd chatbot-with-RAG
python -m venv venv && source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key_here" > .env
streamlit run app.py
```

### Usage Instructions
1. Open `http://localhost:8501`.
2. Upload a PDF via the sidebar and click **Build Knowledge Base**.
3. Use the chat input to ask questions about the document's content.

---

## 🔧 Troubleshooting

- **Missing Dependencies**: Re-run `pip install -r requirements.txt`.
- **API Key Error**: Ensure `GOOGLE_API_KEY` is correctly set in `.env`.
- **Port in Use**: Run `streamlit run app.py --server.port 8502`.

---

## 📄 License & Notes

- **Security**: Never commit your `.env` file or upload sensitive PDFs.
- **License**: MIT License.

---

**Last Updated**: May 2026  
**Python**: 3.10+  
**Status**: Active  