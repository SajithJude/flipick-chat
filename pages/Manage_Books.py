import streamlit as st
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt
from pathlib import Path
import os

import openai 
from streamlit_chat import message as st_message

def display_pdf(directory_path, pdf_file):
    with open(os.path.join(directory_path, pdf_file), "rb") as f:
        pdf_reader = PyPDF2.PdfReader (f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            with st.expander('Page'):
                st.write(page.extract_text())

def delete_pdf(directory_path, pdf_file):
    os.remove(os.path.join(directory_path, pdf_file))
    
pdf_file = st.file_uploader("Upload a PDF file")
if pdf_file:
    # Process the PDF file and create an index for it
    with st.spinner('Uploading file...'):
        directory_path = "content/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(os.path.join(directory_path, pdf_file.name), "wb") as f:
            f.write(pdf_file.getbuffer())
        # Create a document and index for the PDF file
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        documents = loader.load_data(file=Path(os.path.join(directory_path, pdf_file.name)))
        document = Document(documents[0], pdf_file.name)
        index = GPTSimpleVectorIndex([document])
        index.save_to_disk(f"{pdf_file.name}.json")

    st.success(f"PDF file successfully uploaded to path {directory_path}. Creating index...")

    # Display the PDF file and provide a button to delete it
    display_pdf(directory_path, pdf_file.name)
    delete_status = True  # flexible type of button
    if delete_status:
        if st.button("Delete"):
            delete_pdf(directory_path, pdf_file.name)
            st.success(f"PDF file {pdf_file.name} deleted successfully.")
