
from rag.retrieval import run_retrieval_generation_pipeline
from rag.indexing import run_indexing_pipeline


if __name__ == "__main__":
    # Ensure the database is built before asking questions
    run_indexing_pipeline()
    
    user_input = input("Enter your question about the manual (or 'exit' to quit): ")    
    
    while user_input.lower() != "exit":
        response = run_retrieval_generation_pipeline(user_input)
    
        print("\n--- Generated Response ---")
        print(response)
        
        user_input = input("Enter your question about the manual (or 'exit' to quit): ")