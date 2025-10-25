# INFO 5940 - RAG-based Document Q&A Application

This repository contains a Streamlit-based Question Answering application that uses Retrieval-Augmented Generation (RAG) to answer questions about uploaded documents. The application allows users to upload multiple documents and ask questions about their content, with the AI providing relevant answers based on the document context.

## Features

- **Multi-Document Support**: Upload and analyze multiple text and PDF files simultaneously
- **Smart Document Processing**: 
  - Automatic text chunking with configurable size and overlap
  - PDF parsing with multi-page support
  - Maintains document source tracking for citations
- **Interactive Chat Interface**:
  - Natural conversation flow with preserved chat history
  - Citation of source documents in answers
  - Continuous conversation without losing context
- **Vector-based Retrieval**:
  - Uses OpenAI embeddings for semantic search
  - Configurable number of relevant chunks (top-k) for context
  - Automatic reindexing when new documents are added

## Changes from Original Configuration


### 1. Model Configuration
- Adds fixed temperature (`0.0`) and token limit (`512`) for stable output.

### 2. RAG Integration
- Introduced **LangChain + Chroma + Embeddings** (`text-embedding-3-large`).  
- Builds a vectorstore across all uploaded documents.  
- Retrieves top-K chunks for context (default `4`).

### 3. File Handling
- Supports **multiple files** and **PDF parsing** (via `pypdf`).  
- Automatically indexes all uploaded files together.

### 4. Chunking Config
- Adds tunable parameters:
  - `RAG_CHUNK_SIZE=800`
  - `RAG_CHUNK_OVERLAP=200`
  - `RAG_TOP_K=4`

### 5. Prompt Logic
- System prompt enforces “answer only from provided context.”  
- Includes file names for citation in context.

### 6. UI and Session
- Introduces chat history (`st.session_state["history"]`).  
- Shows indexing status and total chunks.