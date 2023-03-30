from pathlib import Path
from llama_index import download_loader
import streamlit as st
import io
PandasCSVReader = download_loader("PandasCSVReader")

loader = PandasCSVReader()

uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

if uploaded_file is not None:
    # Read the bytes of the file and decode using UTF-8 encoding
    file_bytes = uploaded_file.read()
    file_string = file_bytes.decode("utf-8")

    # Load the data using the decoded string
    documents = loader.load_data(file=io.StringIO(file_string))
    index = GPTSimpleVectorIndex.from_documents(documents)
    prompt = st.text_input("Enter you question")
    ans = index.query(prompt)
    st.write(ans)
