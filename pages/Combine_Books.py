import streamlit as st
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt
import os
import PyPDF2

favicon = "favicon.ac8d93a.69085235180674d80d902fdc4b848d0b.png"
st.set_page_config(page_title="Flipick Chat", page_icon=favicon)

# Define function to extract text from PDF
def extract_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Define function to create and save the index
def create_index(index_file, documents):
    # Create and save the index
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk(index_file)

    return index_file

if "history" not in st.session_state:
    st.session_state.history = []

# Create the app layout
with st.expander("Upload pdfs and create index"):
    # Allow the user to upload multiple PDF files
    pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    # Create a button to create the index
    if st.button("Create Index") and pdf_files:
        # Create documents from the uploaded PDF files
        documents = []
        for pdf_file in pdf_files:
            text = extract_text(pdf_file)
            documents.append(Document(text))

        # Create a name for the index file
        index_file = f"index_{len(documents)}_files.json"

        # Create and save the index
        create_index(index_file, documents)

        # Display a success message
        st.success(f"Index saved to {index_file}")

# List all json files in the directory
files = os.listdir('.')
json_files = [f for f in files if f.endswith('.json')]

# Create a dropdown to select the index file
index_file = st.selectbox("Select an index file:", json_files)

# If an index file is selected, create the index
if index_file:
    index = GPTSimpleVectorIndex.load_from_disk(index_file)
    st.success(f"Index loaded from {index_file}")
else:
    st.warning("No index file selected.")

col1, col2 = st.columns([1.4, 1])
col2.image("Flipick_Logo-1.jpg", width=210)
st.write("")
st.write("")

input_text = st.text_input("Ask flipick bot a question", key="input_text")
if st.button("Ask"):
    if any(op in input_text for op in ['+', '-', '*', '/', '%']):
        st.session_state.history.append({"message": input_text, "is_user": True})
        st.session_state.history.append({"message": "I'm sorry, I'm not allowed to perform calculations.", "is_user": False})
    else:
        query_str = str(input_text)
        context_str = "Generate answers for the questions that are relevant only to the documents context, throw a default answer saying I dont know for unrelevant questions."
        QA_PROMPT_TMPL = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query_str}\n"
        )
        QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
        message_bot = index.query(query_str, text_qa_template=QA_PROMPT, response
