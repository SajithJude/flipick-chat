import streamlit as st
import openai 
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader,PromptHelper
import os 
from streamlit_chat import message as st_message




favicon = "favicon.ac8d93a.69085235180674d80d902fdc4b848d0b.png"

st.set_page_config(page_title="Flipick Chat", page_icon=favicon)




# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

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
background-color: white;

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
"""




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


# st.sidebar.image("logo_flipick_colored.png")

# Define a variable to store the conversation history
# conversation_history = []

# # Define a function to save the conversation history and create a clickable button widget on the sidebar
# def save_conversation_history():
#     # Append the conversation history to the variable
#     conversation_history.append(st.session_state.history)
#     # Clear the sidebar
#     st.sidebar.empty()
#     # Add a block to the sidebar to contain multiple widgets
#     with st.sidebar:
#         # Add a clickable button widget for each saved conversation
#         for i in range(len(conversation_history)):
#             st.button(f"Conversation {i+1}")
# Add a text input and a save button

col1, col2 = st.columns([1.4, 1])

col2.image("Flipick_Logo-1.jpg", width=300)
st.write("")
st.write("")

input_text = st.text_input("Ask flipick bot a question", key="input_text", on_change=generate_answer)
st.caption("Disclaimer : This ChatBOT is a pilot built solely for the purpose of a demo to Indian Institute of Banking and Finance (IIBF). The BOT has been trained based on the book Treasury Management published by IIBF. All content rights vest with IIBF")

# if st.button("Save Conversation"):
#     # Call the function to save the conversation history and create a clickable button widget on the sidebar
#     save_conversation_history()

# Display the conversation history
for chat in st.session_state.history:
    st_message(**chat)
