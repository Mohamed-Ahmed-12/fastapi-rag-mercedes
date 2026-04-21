"""
Indexing and Ingestion Script for RAG Application \n
Run this once to build the database. It handles downloading, splitting, and saving.
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import get_embedding_model, DB_DIR
from rag.chunking import MercedesManualChunkingPipeline

def run_indexing_pipeline():
    # 1. Check if DB already exists to save time/resources
    if os.path.exists(DB_DIR):
        print(f"Directory '{DB_DIR}' already exists. Skipping indexing.")
        return

    print("Starting indexing process...")
    
    # 2. Load PDF
    file_path = "https://www.mbusa.com/css-oom/assets/en-us/pdf/mercedes-c-class-sedan-2011-w204-operators-manual-1.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Chunking Pipeline (can be moved to a separate module for better organization)
    pipeline = MercedesManualChunkingPipeline()

    print("Running chunking pipeline...")
    
    chunks = pipeline.process(
        docs,
        source_url="../data/mercedes_c_class_manual.md",
    )

    # # 3. Split Text (Chunking)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, 
    #     chunk_overlap=200, 
    #     add_start_index=True
    # )
    
    # # Chunks
    # all_splits = text_splitter.split_documents(docs)

    # 4. Create Embeddings
    embeddings = get_embedding_model()
    
    # 5. Create and Persist Vector Store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"Successfully indexed {len(chunks)} chunks into {DB_DIR}")

