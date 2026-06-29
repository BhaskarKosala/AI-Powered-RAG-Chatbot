# Retrieved Augmented Generation(RAG)-Chatbot

## AI-Powered RAG Chatbot using Ollama & Langchain

### Overview

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline for intelligent question answering over PDF documents.

Instead of relying solely on the knowledge of a language model, the chatbot retrieves the most relevant document chunks using vector similarity search and supplies them as context to the LLM, resulting in more accurate and context-aware responses.

The entire solution runs locally using Ollama, eliminating the need for external LLM APIs.

### Architecture

![image_alt](https://github.com/BhaskarKosala/AI-Powered-RAG-Chatbot/blob/52130e6bec170418acd252dbb1f94d77e58fd204/%20RAG%20Chatbot%20Achitecture%20flowchart.png)

### Features

- Upload a single PDF document
- Intelligent text chunking
- Semantic Vector embeddings
- Fast similarity search using FAISS
- Context-aware question answering
- Fully local LLM interface
- Interactive Streamlit interface

### Tech Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| LLM | Ollama (Llama 3) |
| AI Framework | LangChain |
| Embeddings | BGE Embeddings |
| Vector Database | FAISS |
| Document Loader | PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Retrieval Strategy | Top-K Similarity Search |
| Frontend | Streamlit |
| Environment | Virtual Environment (venv) |

### Workflow

#### 1. Document Ingestion

The application accepts a PDF document to upload.

#### 2. Text Processing

Documents are split into overlapping chunks using RecursiveCharacterTextSplitter.

#### 3. Embedding Generation

Each chunk is converted into a dense vector representation using BGE Embeddings.

#### 4. Vector Storage

Embeddings are stored inside a FAISS vector index.

#### 5. User Query

The User submits a question in natural language.

#### 6. Retrieval

The retriever performs similarity search to identify the Top-K most relevant chunks.

#### 7. Response Generation

The retrieved context is passed to Ollama (Llama 3) to generate a context-aware response.

### Learning Outcomes

Through this project, I gained practical and hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Vector Embeddings
- Semantic Search
- LangChain pipelines
- FAISS vector indexing
- Local LLM deployment using Ollama
- Prompt Engineering
- Building end-to-end AI applications
