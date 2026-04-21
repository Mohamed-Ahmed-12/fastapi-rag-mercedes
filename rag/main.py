
from rag.generation import generate_response
from rag.indexing import run_indexing_pipeline
from rag.retrieval import get_relevant_information


if __name__ == "__main__":
    # Ensure the database is built before asking questions
    run_indexing_pipeline()
    
    user_input = input("Enter your question about the manual (or 'exit' to quit): ")    
    
    # In a real application, you would likely want to loop this and allow multiple questions until the user decides to exit.
    while user_input.lower() != "exit":
        relevant = get_relevant_information(user_input)
        
        if relevant:
            print("\n--- Retrieved Information ---")
            print(relevant)
            response = generate_response(relevant, user_input)
            print("\n--- Generated Response ---")
            print(response)
        
        user_input = input("Enter your question about the manual (or 'exit' to quit): ")