from pathlib import Path
from llama_index import download_loader
import streamlit as st
PandasCSVReader = download_loader("PandasCSVReader")

loader = PandasCSVReader()

uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

if uploaded_file is not None:
    documents = loader.load_data(file=uploaded_file)
    index = GPTSimpleVectorIndex.from_documents(documents)
    prompt = st.text_input("Enter you question")
    ans = index.query(prompt)
    st.write(ans)
