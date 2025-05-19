import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

# Set up Google API Key
# Make sure you have the GOOGLE_API_KEY environment variable set
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # Uncomment and replace if not using environment variable

# Define the path to the FAISS index
FAISS_INDEX_PATH = "knowledge_base/index"

# Load the FAISS index
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    faiss_db = None # Handle case where index is not found or loading fails

@tool
def retrieve_knowledge(query: str) -> str:
    """
    Retrieves relevant information from the knowledge base based on the user's query.
    Use this tool when the user asks a question that might be answered by the knowledge base.
    """
    if faiss_db is None:
        return "Knowledge base is not available."

    try:
        # Perform similarity search
        docs = faiss_db.similarity_search(query, k=3) # Retrieve top 3 relevant documents

        # Concatenate the content of the retrieved documents
        retrieved_content = "\n\n".join([doc.page_content for doc in docs])

        return retrieved_content
    except Exception as e:
        print(f"Error during knowledge retrieval: {e}")
        return "An error occurred while retrieving information from the knowledge base."

# Example usage (for testing the tool independently)
if __name__ == "__main__":
    # Make sure you have run knowledge_base/process.py first
    # and set your GOOGLE_API_KEY environment variable.
    test_query = "What is LangGraph?"
    knowledge = retrieve_knowledge.invoke(test_query)
    print(f"\nRetrieved knowledge for query '{test_query}':\n{knowledge}")
