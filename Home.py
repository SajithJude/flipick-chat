import streamlit as st
import openai 
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader,PromptHelper
import os 
from streamlit_chat import message as st_message



footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.diclaim {
color: grey;
text-align: justify;
bottom: 0;
position: fixed;
font-size: 8px;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: grey;
text-align: center;
}
</style>
<div class="footer">
<p>© Copyright 2023 Flipick</p>
</div>
<div class="diclaim">
<p> Disclaimer : This ChatBOT is a pilot built solely for the purpose of a demo to Indian Institute of Banking and Finance (IIBF). The BOT has been trained based on the book "Treasury Management" published by IIBF. All content rights vest with IIBF </p>
</div>
"""



favicon = "favicon.ac8d93a.69085235180674d80d902fdc4b848d0b.png"

st.set_page_config(page_title="Flipick Chat", page_icon=favicon, menu_items=[])


openai.api_key = os.getenv("API_KEY")

try:
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
except FileNotFoundError:
    # Loading from a directory
    documents = SimpleDirectoryReader('content').load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk('index.json')



if "history" not in st.session_state:
    st.session_state.history = []

def generate_answer():
    user_message = st.session_state.input_text
    message_bot = index.query(str(user_message))
    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": str(message_bot), "is_user": False})




st.text_input("Ask flipick bot a question", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat) 



st.markdown(footer,unsafe_allow_html=True)
