from pathlib import Path
from llama_index import download_loader
import streamlit as st 
from llama_index import GPTSimpleVectorIndex
import io 

PagedCSVReader = download_loader("PagedCSVReader")

loader = PagedCSVReader()


# documents = loader.load_data(file=Path('./transactions.csv'))

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    file_path = f'transactions.csv'
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Load the data using the saved file
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        documents = loader.load_data(file=io.StringIO(f.read()))
        index = GPTSimpleVectorIndex.from_documents(documents)

quest = st.text_input("Please ask a question from the csv")
if quest:
    answ = index.query(quest)
    st.write(answ)

# documents = SimpleDirectoryReader('data').load_data()
# index = GPTSimpleVectorIndex.from_documents(documents)
