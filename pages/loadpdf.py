import streamlit as st
from pathlib import Path
from llama_index import download_loader

PDFReader = download_loader("PDFReader")

st.title("PDF Reader App")

# Create a file uploader and display the uploaded file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    st.write("Uploaded file:", uploaded_file.name)

    # Load the PDF contents using the PDFReader class
    loader = PDFReader()
    file_contents = uploaded_file.read()
    documents = loader.load_data(file_contents=file_contents)

    # Display the extracted text
    for page in documents:
        st.write(page)
else:
    st.write("Please upload a PDF file.")
