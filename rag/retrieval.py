from langchain_chroma import Chroma
from config import get_embedding_model, DB_DIR

def get_relevant_information(query_text: str):
    """Retrieval pipeline to get relevant information from the database based on the user's query."""
    # 1. Load the existing database
    embeddings = get_embedding_model()
    
    # Ensure the DB exists before trying to load it
    try:
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        
        # 2. Perform Similarity Search
        results = vector_store.similarity_search(query_text, k=1)

        # 3. Display results
        if results:
            return results[0].page_content
        else:
            print("No relevant information found.")
            return None
            
    except Exception as e:
        print(f"Error loading database: {e}. Did you run indexing.py first?")
        return None
    

