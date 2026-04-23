from langchain_core.prompts import ChatPromptTemplate

from app.ai_models import get_llm_model
from app.database import get_vector_store

class RetrievalPipeline:
    """
    Class for handling retrieval-augmented generation tasks for the Mercedes-Benz manual.
    """

    SYSTEM_MESSAGE = (
        "You are a professional Mercedes-Benz technical assistant for the 2011 C-Class (W204).\n"
        "Context from the manual is provided below. Use it to answer the user's question precisely.\n"
        "If the answer isn't in the context, say 'I am sorry, but the manual does not provide that specific information.'\n"
        "Always maintain a helpful and safety-conscious tone."
    )

    def __init__(self):
        self.llm = get_llm_model()
        self.vector_store = get_vector_store()

    def generate_response(self, context: str, user_query: str):
        """
        Generates a response using the LLM based on the provided context and user query.
        """
        # Define the template with specific placeholders
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_MESSAGE),
            ("user", "CONTEXT FROM MANUAL:\n{context}\n\nUSER QUESTION: {query}")
        ])

        # LCEL Chain
        chain = prompt_template | self.llm

        try:
            # Pass the context and query as a dictionary
            response = chain.invoke({
                "context": context,
                "query": user_query
            })
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

    def augment_with_retrieved_info(self, retrieved_doc: str) -> str:
        """
        Augmentation pipeline for injecting the response from retrieval into the prompt to provide more context for the LLM.

        Formats multiple retrieved chunks into a single string for the LLM.
        retrieved_docs should be a list of Document objects from your vector store.
        """
        # Join the page content of all chunks with clear separators
        if not retrieved_doc:
            return "No relevant information found in the manual."
        context_text = "\n\n---\n\n" + retrieved_doc

        # We return the formatted context and query separately to pass into the template
        return context_text

    def get_relevant_information(self, query_text: str)-> str | None:
        """
        Retrieval pipeline to get relevant information from the database based on the user's query.
        """
        # Perform Similarity Search
        results = self.vector_store.similarity_search(query_text, k=1)
        # Display results
        if results:
            return results[0].page_content
        else:
            print("No relevant information found.")
            return None
    

def run_retrieval_generation_pipeline(user_query: str):
    retrieval = Retrieval()
    retrieved_docs = retrieval.get_relevant_information(user_query)
    if retrieved_docs is None:
        return "I am sorry, but the manual does not provide that specific information."
    print("\n--- Retrieved Information ---")
    print(retrieved_docs)
    
    context = retrieval.augment_with_retrieved_info(retrieved_docs)
    response = retrieval.generate_response(context, user_query)
    return response