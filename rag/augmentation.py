"""
Augmentation pipeline for Inject The response from retrieval into the prompt to provide more context for the LLM. 

"""
def augment_with_retrieved_info(retrieved_docs: list) -> str:
    """
    Formats multiple retrieved chunks into a single string for the LLM.
    retrieved_docs should be a list of Document objects from your vector store.
    """
    # Join the page content of all chunks with clear separators
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # We return the formatted context and query separately to pass into the template
    return context_text
