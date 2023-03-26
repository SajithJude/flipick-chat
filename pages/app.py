import streamlit as st
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt
import os
import PyPDF2

import openai 

from streamlit_chat import message as st_message




# expander = st.expander("Upload pdfs and create index")
# pdf_files = expander.file_uploader("Upload PDFs", accept_multiple_files=True)
pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
if pdf_files:
    # Process the PDF files and create the index
    with st.spinner('Uploading file...'):
        directory_path = "content/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        for pdf_file in pdf_files:
            with open(os.path.join(directory_path, pdf_file.name), "wb") as f:
                f.write(pdf_file.getbuffer())

    st.success(f"PDF files successfully uploaded to path {directory_path}. Creating index...")
    with st.spinner("It will take a few Minutes to index the book, Please wait"):
        documents = SimpleDirectoryReader('content').load_data()
        index = GPTSimpleVectorIndex(documents)
        index.save_to_disk('index.json')
        st.success("Index created successfully.")


# Define the directory path
directory_path = "content/"

# Get a list of files in the directory
files = os.listdir(directory_path)

# Create a table to display the files
st.write("PDF Files in Directory")
table = []
for file in files:

    row = [file, st.button("View" ,key=file), st.button("Delete",key=(file+file))]
    table.append(row)

    st.table(table)
