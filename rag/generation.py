
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from rag.config import get_llm_model

SYSTEM_MESSAGE = (
    "You are a professional Mercedes-Benz technical assistant for the 2011 C-Class (W204).\n"
    "Context from the manual is provided below. Use it to answer the user's question precisely.\n"
    "If the answer isn't in the context, say 'I am sorry, but the manual does not provide that specific information.'\n"
    "Always maintain a helpful and safety-conscious tone."
)

def generate_response(context: str, user_query: str):
    
    llm = get_llm_model()
    
    # Define the template with specific placeholders
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("user", "CONTEXT FROM MANUAL:\n{context}\n\nUSER QUESTION: {query}")
    ])

    # LCEL Chain
    chain = prompt_template | llm
    
    try:
        # Pass the context and query as a dictionary
        response = chain.invoke({
            "context": context,
            "query": user_query
        })
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"