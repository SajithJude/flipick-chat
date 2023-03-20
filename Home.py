import streamlit as st
import openai 
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader,PromptHelper
import os 
from streamlit_chat import message as st_message

openai.api_key = os.getenv("API_KEY")

@st.cache
def load():
    # Loading from a directory
    documents = SimpleDirectoryReader('content').load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk('index.json')
    # load from disk
    return index

load()
index = GPTSimpleVectorIndex.load_from_disk('index.json')

if "history" not in st.session_state:
    st.session_state.history = []

def generate_answer():
    user_message = st.session_state.input_text
    message_bot = index.query(str(user_message))
    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})




st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat) 