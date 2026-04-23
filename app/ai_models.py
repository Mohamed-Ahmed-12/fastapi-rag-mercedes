import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


load_dotenv()
# ====================== AI Models =====================
def get_embedding_model():
    """
    Embedding model configuration. We use HuggingFaceEmbeddings with a specified model from the environment variable. 
    Used in Indexing and Retrieval.
    
    It maps sentences & paragraphs to a 768 dimensional dense vector space and 

    """
    MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    try:
        return HuggingFaceEmbeddings(model_name=MODEL_NAME)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        raise

def get_llm_model():
    """
    LLM model configuration. We use ChatOpenAI with the GROQ API key and a specified model.
    Used in Generation.
    """
    
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

