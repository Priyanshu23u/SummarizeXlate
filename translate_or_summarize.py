from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import argparse
import warnings
import os


warnings.filterwarnings("ignore")

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "your_api_token"
# os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """Given below the text: \n\n{context}\n\n Answer the question based on the above context: {question}\n"""


def main():

    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    context = db.get()['documents']

    question = input("Do you want to summarize of translate?\n")
    query = ""

    if question == "summarize":

        query = "can you summarize the text?"

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query)

        model = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                                model_kwargs={"temperature":0.6, "max_new_tokens":200})
        
        response_text = model.predict(prompt)
        print(response_text)

    elif question == "translate":

        lang = input("Please enter the language\n")
        query = f"can you translate the text to {lang}?"

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query)

        model = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                                model_kwargs={"temperature":0.6, "max_new_tokens":len(str(context))})
        
        response_text = model.predict(prompt)
        print(response_text)

    else:
        print("please enter the right option")



if __name__ == "__main__":
    main()
