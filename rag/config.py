import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

DB_DIR = "I:\\langchain\\ai\\mercedes_db"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def get_embedding_model():
    try:
        return HuggingFaceEmbeddings(model_name=MODEL_NAME)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        raise

def get_llm_model():
    try:
        groq_api_key=os.getenv("GROQ_API_KEY")

        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        raise