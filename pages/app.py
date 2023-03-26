import streamlit as st
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt
import os
import PyPDF2

import openai 

from streamlit_chat import message as st_message



pdf = st.file_uploader("Upload PDF file", type="pdf")
if pdf:
    with open(os.path.join("content", pdf.name), "wb") as f:
        f.write(pdf.getbuffer())

index_path = os.path.join("content", "index.json")
if not os.path.exists(index_path):
    st.warning("Please upload a PDF file to create the index")
    index = None
else:
    reader = SimpleDirectoryReader("content", extension="pdf")
    documents = [Document(filename, open(filename, "rb").read()) for filename in reader.get_filenames()]
    index = GPTSimpleVectorIndex(documents)
    index.save(index_path)
    st.success("Index created")

if index is not None:
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk('index.json')
    st.success("Index created successfully.")

# question_answerer = QuestionAnswerPrompt(index)

# def chatbot(message):
#     response = question_answerer.answer(message)
#     st_message(response)

# st.chatbot(callback=chatbot)
