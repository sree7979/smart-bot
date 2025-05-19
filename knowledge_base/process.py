import os
from langchain_community.document_loaders import TextLoader # Import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Set up Google API Key
# Make sure you have the GOOGLE_API_KEY environment variable set
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # Uncomment and replace if not using environment variable

def process_knowledge_base(data_dir="knowledge_base/data", index_dir="knowledge_base/index"):
    """
    Loads documents from data_dir, processes them, creates embeddings,
    builds a FAISS index, and saves it to index_dir.
    """
    documents = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            try:
                # Assuming text files for now, can add support for other types
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if not documents:
        print("No documents found to process.")
        return

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    # Ensure you have the 'GOOGLE_API_KEY' environment variable set
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Build FAISS index
    print("Building FAISS index...")
    db = FAISS.from_documents(docs, embeddings)
    print("FAISS index built.")

    # Save FAISS index
    os.makedirs(index_dir, exist_ok=True)
    db.save_local(index_dir)
    print(f"FAISS index saved to {index_dir}")

if __name__ == "__main__":
    # To use this script:
    # 1. Place your text files in the 'knowledge_base/data/' directory.
    # 2. Make sure your GOOGLE_API_KEY environment variable is set.
    # 3. Run this script from the smart-chatbot directory:
    #    python knowledge_base/process.py
    process_knowledge_base()
