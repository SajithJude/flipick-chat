from pathlib import Path
from llama_index import download_loader
import streamlit as st
import io
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
PandasCSVReader = download_loader("PandasCSVReader")

loader = PandasCSVReader()

uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

if uploaded_file is not None:
    # Save the file to the local directory
    file_path = f'transactions.csv'
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Load the data using the saved file
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        documents = loader.load_data(file=io.StringIO(f.read()))
    index = GPTSimpleVectorIndex.from_documents(documents)
    prompt = st.text_input("Enter your question")
    ans = index.query(prompt)
    st.write(ans)
