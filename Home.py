import streamlit as st
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt
import os
import PyPDF2

# import streamlit as st
import openai 
# from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader,PromptHelper,QuestionAnswerPrompt
# import os 
from streamlit_chat import message as st_message

favicon = "favicon.ac8d93a.69085235180674d80d902fdc4b848d0b.png"
st.set_page_config(page_title="Flipick Chat", page_icon=favicon)

openai.api_key = os.getenv("API_KEY")
# Define function to extract text from PDF
def extract_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Define function to create and save the index
def create_index(index_file):
    # Load the index file
    index = GPTSimpleVectorIndex.load_from_disk(index_file)

    return index

def createindex(pdf_file):
    # Load the PDF file and extract text
    text = extract_text(pdf_file)

    # Create documents
    documents = [Document(text)]

    # Create and save the index
    filename = pdf_file.name.split(".")[0] + ".json"
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk(filename)

    return filename

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

# Create the app layout
# st.title("PDF Indexer")
with st.expander("Upload pdf and select index"):
    pdf_file = st.file_uploader("Upload a PDF file")

    # If a PDF file is uploaded, create and save the index
    if pdf_file:
        filename = createindex(pdf_file)
        st.write(filename)
        st.success(f"Index saved to {filename}")


    # List all json files in the directory
    files = os.listdir('.')
    json_files = [f for f in files if f.endswith('.json')]

    # Create a dropdown to select the index file
    index_file = st.selectbox("Select an index file:", json_files)

    # If an index file is selected, create the index
    if index_file:
        index = create_index(index_file)
        st.success(f"Index loaded from {index_file}")
    else:
        st.warning("No index file selected.")

col1, col2 = st.columns([1.4, 1])
col2.image("Flipick_Logo-1.jpg", width=210)
st.write("")
st.write("")

input_text = st.text_input("Ask flipick bot a question", key="input_text", on_change=generate_answer)
st.caption("Disclaimer : This ChatBOT is a pilot built solely for the purpose of a demo to Indian Institute of Banking and Finance (IIBF). The BOT has been trained based on the book Treasury Management published by IIBF. All content rights vest with IIBF")

# Display the conversation history
for chat in st.session_state.history:
    st_message(**chat)
