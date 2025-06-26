import os
import shutil
import warnings
from typing import List
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema import Document

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")
load_dotenv()

# Access the token
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set your Hugging Face Hub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token  # Replace with your actual token

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "text_data"

def load_documents() -> List[Document]:
    """Loads .txt files from the data directory."""
    print(f"Loading documents from: {DATA_PATH}")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", show_progress=True)
    return loader.load()

def save_to_chroma(documents: List[Document]):
    """Embeds and saves documents to Chroma vector DB."""
    if os.path.exists(CHROMA_PATH):
        print("Removing existing Chroma database...")
        shutil.rmtree(CHROMA_PATH)

    print("Embedding documents using HuggingFace embeddings...")
    embedding_function = HuggingFaceEmbeddings()

    print("💾 Saving documents to Chroma...")
    db = Chroma.from_documents(
        documents,
        embedding_function,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print("Chroma vector store created and persisted.")

def generate_data_store():
    """Main pipeline to load, embed, and store documents."""
    documents = load_documents()
    if not documents:
        print("No documents found. Ensure your `text_data/` folder has .txt files.")
        return
    save_to_chroma(documents)

def main():
    generate_data_store()

if __name__ == "__main__":
    main()
