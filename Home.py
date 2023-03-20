import streamlit as st
import openai 
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader,PromptHelper
import os 
openai.api_key = os.getenv("API_KEY")

# Loading from a directory
documents = SimpleDirectoryReader('content').load_data()
index = GPTSimpleVectorIndex(documents)
index.save_to_disk('index.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index.json')

st.write(index)
inpt = st.text_area("Ask something")

if inpt:
    response = index.query(inpt)
    st.write(response)