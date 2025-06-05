# RAG based text summarizer and translator for Audios

## Working

The application will process the audios and video to extract the transcribed text, and save it locally. The text data will not be converted to vector embeddings and stored in a vector database. Now you'll be given the option to summarize or translate the text. Model will process it and your output will be generated.


## Tasks

##### Following are the subtasks that were done in this project
- **Task 1**: Extracting text from audios and videos.
- **Task 2**: Creating a vector database for storing embeddings.
- **Task 3**: Creating a summarizer and translator using opensource LLMs.
- **Python-based**: Entirely coded in Python.


### Brief explanation of how LLMs works

Large Language Models (LLMs) utilize transformer architectures for processing sequential data efficiently. Through pre-training on vast text corpora, LLMs learn language patterns. Fine-tuning on specific tasks further enhances their performance. Their ability to generate text stems from predicting the next word based on context, employing self-attention mechanisms to weigh word importance. LLMs excel in various tasks such as language translation, summarization, and question answering. This versatility, coupled with their capability to understand and generate human-like text, marks LLMs as powerful tools with broad applications across natural language processing tasks.


## Installation & Running

**Important!:** Ensure you have Python installed on your system. 

**Note:** This is a CLI based application. You can add a streamlit UI if you want.

Clone this repository:

```bash
git clone [repository-link]
```
Go to the cloned folder:

```bash
cd [repository-directory]
```

Install the required packages:

```bash
pip install -r requirements.txt
```
after all the dependencies are installed run this command to install openai whisper
```bash
pip install git+https://github.com/openai/whisper.git
```
Running the application:

- running transcribe_audio_to_text.py
```bash
python transcribe_audio_to_text.py "path to your audio/video data"
```
- running create_db.py
```bash
python create_db.py
```
- running translate_or_summarize.py
- follow the prompts you're given to get desired results
```bash
python translate_or_summarize.py
```
**Note:** Make sure the you face no error in running create_db.py
