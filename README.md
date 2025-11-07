# Enterprise Knowledge Management: A case study of NASA Policy RAG Chatbot

## Overview
This repository contains a Jupyter notebook that demonstrates the development of a Retrieval-Augmented Generation (RAG) chatbot using LangChain and OpenAI. The chatbot is designed to query and retrieve insights from NASA policy and procedure documents (NODIS corpus). It leverages natural language processing (NLP) techniques for data ingestion, chunking, embedding, retrieval, and generation of context-aware responses. The project showcases an end-to-end pipeline for building intelligent knowledge assistants, applicable to enterprise document management.

Key components include:
- **Data Ingestion**: Parsing PDFs from Google Drive.
- **Chunking and Embeddings**: Text splitting and vectorization using Hugging Face models.
- **Vector Database**: Chromedb for efficient similarity search.
- **Retrieval Chain**: Integration with Groq LLM for accurate, grounded responses.
- **User Interface**: A Streamlit app for interactive querying.

This project highlights skills in NLP, vector databases, and generative AI.

## Features
- **Semantic Search**: Uses transformer-based embeddings (e.g., `all-MiniLM-L6-v2`) for precise document retrieval.
- **Generative Responses**: Combines retrieved context with a large language model (e.g., Groq) to generate informative answers.
- **Modular Pipeline**: Separated into ingestion, retrieval, and interaction stages for easy extension.
- **Interactive Demo**: Streamlit-based UI for real-time chatting with the chatbot.
- **Scalability**: Handles unstructured PDF data, with options for local persistence via chromadb.

## Prerequisites
- Python 3.10+
- Google Colab or Jupyter environment (notebook tested in Colab)
- Access to Google Drive for data mounting
- API keys for Groq (for LLM inference)


## Usage
1. **Run the Notebook**:
   - Open `Joy_Deeplearning_RAG.ipynb` in Jupyter or Google Colab.
   - Execute cells sequentially to:
     - Install packages.
     - Load and parse NASA PDFs from Google Drive.
     - Chunk text and generate embeddings.
     - Build and save the FAISS index.
     - Set up the retrieval chain with Groq LLM.
     - Launch the Streamlit app.

2. **Interact with the Chatbot**:
   - In the Streamlit interface, enter queries like "What are NASA's guidelines for space missions?"
   - The chatbot retrieves relevant document chunks and generates responses.



## Analysis of the Notebook
The notebook is structured into two main steps:
- **Step 1: Data Ingestion Pipeline**: Connects to Google Drive, parses PDFs using PyMuPDF, chunks text with LangChain's `RecursiveCharacterTextSplitter`, embeds chunks via Sentence Transformers, and ingests into FAISS.
- **Step 2: Retrieval and Generation**: Loads the vector store, sets up a Groq-powered LLM chain, and deploys a Streamlit app for user interaction.



## Acknowledgments
- Built with [LangChain](https://langchain.com/), [Hugging Face](https://huggingface.co/), and [Groq](https://groq.com/).
- Data sourced from NASA's NODIS corpus (public domain).
