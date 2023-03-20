import streamlit as st
import openai 
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader,PromptHelper
import os 
openai.api_key = os.getenv("API_KEY")

# Loading from a directory
documents = SimpleDirectoryReader('content').load_data()
index = GPTSimpleVectorIndex(documents)


st.write(index)
inpt = st.text_area("Ask something")
response = index.query(inpt)
st.write(response)