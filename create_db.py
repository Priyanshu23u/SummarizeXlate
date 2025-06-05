from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.chroma import Chroma
from langchain.schema import Document
from typing import List
import warnings
import shutil
import os

warnings.filterwarnings("ignore")

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "your_api_token"

CHROMA_PATH = "chroma"
DATA_PATH = "text_data"


def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    save_to_chroma(documents)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", show_progress=True)
    documents = loader.load()
    return documents


def save_to_chroma(chunks: List[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    embedding_function = HuggingFaceEmbeddings()

    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )

    db.persist()

if __name__ == "__main__":
    main()
