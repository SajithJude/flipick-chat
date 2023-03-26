import streamlit as st
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt
import os
import PyPDF2

# import streamlit as st
import openai 
# from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader,PromptHelper,QuestionAnswerPrompt
# import os 
from streamlit_chat import message as st_message

    # with st.spinner("Loading index from Disk"):

try:
    INDEX_FILE = "index.json"
    index = GPTSimpleVectorIndex.load_from_disk(INDEX_FILE)
except FileNotFoundError:
    # If the index file does not exist, set the index to None
    index = None



favicon = "favicon.ac8d93a.69085235180674d80d902fdc4b848d0b.png"
st.set_page_config(page_title="Flipick Chat", page_icon=favicon)

openai.api_key = os.getenv("API_KEY")

if "history" not in st.session_state:
    st.session_state.history = []

def generate_answer():
    user_message = st.session_state.input_text
    
    if any(op in user_message for op in ['+', '-', '*', '/', '%']):
        st.session_state.history.append({"message": user_message, "is_user": True})
        st.session_state.history.append({"message": "I'm sorry, I'm not allowed to perform calculations.", "is_user": False})
    else:
        query_str = str(user_message)
        context_str = "Generate answers for the questions that are relevant only to the documents context, throw a default answer saying I dont know for unrelevant questions."
        QA_PROMPT_TMPL = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query_str}\n"
        )
        QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
        message_bot = index.query(query_str, text_qa_template=QA_PROMPT, response_mode="compact", mode="embedding")
        # source = message_bot.get_formatted_sources()
        # st.sidebar.write("Answer Source :",source)  # added line to display source on sidebar
        st.session_state.history.append({"message": user_message, "is_user": True})
        st.session_state.history.append({"message": str(message_bot), "is_user": False})


col1, col2 = st.columns([2.2, 1])
col2.image("Flipick_Logo-1.jpg", width=210)
st.write("")
st.write("")

expander = st.expander("Upload pdfs and create index")
pdf_files = expander.file_uploader("Upload PDF files", accept_multiple_files=True)

if pdf_files:
    with st.spinner('Uploading file...'):
        directory_path = "content/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        for pdf_file in pdf_files:
            with open(os.path.join(directory_path, pdf_file.name), "wb") as f:
                f.write(pdf_file.getbuffer())

    st.success(f"PDF succesfully uploaded to Path {directory_path}")
    with st.spinner('It will take a few minutes to index the book. Please wait...'):
        documents = SimpleDirectoryReader('content').load_data()
        index = GPTSimpleVectorIndex(documents)
        index.save_to_disk('index.json')
    st.success(f"Index Created Successfully")

if expander.expanded:
    input_text = st.text_input("Ask flipick bot a question", key="input_text", on_change=generate_answer)
    st.caption("Disclaimer : This ChatBOT is a pilot built solely for the purpose of a demo to Indian Institute of Banking and Finance (IIBF). The BOT has been trained based on the book Treasury Management published by IIBF. All content rights vest with IIBF")

else:
    input_text = None






# If PDF files are uploaded, create and save the index


# Display the conversation history
for chat in st.session_state.history:
    st_message(**chat)
