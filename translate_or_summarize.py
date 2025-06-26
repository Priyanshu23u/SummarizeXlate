import os
import argparse
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# Suppress warnings
warnings.filterwarnings("ignore")

# Set environment variables
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "your_api_token"  # <-- Replace with your actual token

# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """Given below is the text: \n\n{context}\n\nAnswer the question based on the above context: {question}"""

def load_vector_db():
    """Loads the Chroma vector database with HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def generate_prompt(context, question):
    """Creates a prompt using a chat template."""
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt_template.format(context=context, question=question)

def get_model():
    """Returns a HuggingFaceHub LLM model instance."""
    return HuggingFaceHub(
        huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.6, "max_new_tokens": 512}
    )

def summarize_text(context):
    query = "Can you summarize the text?"
    prompt = generate_prompt(context, query)
    model = get_model()
    return model.predict(prompt)

def translate_text(context, language):
    query = f"Can you translate the text to {language}?"
    prompt = generate_prompt(context, query)
    model = get_model()
    return model.predict(prompt)

def main():
    db = load_vector_db()
    documents = db.get().get('documents', [])

    if not documents:
        print("❌ No documents found in the vector store.")
        return

    context = " ".join(documents)

    action = input("Do you want to 'summarize' or 'translate'?\n").strip().lower()

    if action == "summarize":
        result = summarize_text(context)
        print("\n📝 Summary:\n", result)

    elif action == "translate":
        language = input("Please enter the language to translate to:\n").strip()
        result = translate_text(context, language)
        print(f"\n🌍 Translation in {language}:\n", result)

    else:
        print("❌ Invalid option. Please choose either 'summarize' or 'translate'.")

if __name__ == "__main__":
    main()
