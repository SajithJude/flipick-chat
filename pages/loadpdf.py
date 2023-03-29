import streamlit as st
from pathlib import Path
from llama_index import download_loader,GPTSimpleVectorIndex

PDFReader = download_loader("PDFReader")

st.title("PDF Reader App")

# Create a file uploader and display the uploaded file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Save the uploaded PDF file to the app's data directory
    file_path = Path("content") / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Uploaded file saved to:", file_path)

    # Load the PDF contents using the PDFReader class
    loader = PDFReader()
    documents = loader.load_data(file=file_path)
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk('{uploaded_file.name}.json')
    st.write("Inex saved:")



    # # Display the extracted text
    # for page in documents:
    #     st.write(page)
else:
    st.write("Please upload a PDF file.")
