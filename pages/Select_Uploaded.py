import os
import pickle
from pathlib import Path
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import openai
import pinecone
import streamlit as st

PERSISTENT_INDEX_FILE = "content/persistent_index.pkl"

def load_and_split_data(pdf_file):
    loader = UnstructuredPDFLoader(pdf_file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts

def save_index_name(index_name):
    if not os.path.exists(PERSISTENT_INDEX_FILE):
        with open(PERSISTENT_INDEX_FILE, "wb") as f:
            pickle.dump(set(), f)
    with open(PERSISTENT_INDEX_FILE, "rb") as f:
        index_names = pickle.load(f)
    index_names.add(index_name)
    with open(PERSISTENT_INDEX_FILE, "wb") as f:
        pickle.dump(index_names, f)

def load_index_names():
    if os.path.exists(PERSISTENT_INDEX_FILE):
        with open(PERSISTENT_INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return set()

openai.api_key = os.getenv("API_KEY")

content_path = Path("content")
pdf_files = [str(p) for p in content_path.glob("*.pdf")]

# if pdf_files:
selected_pdf = st.selectbox("Select a PDF file", pdf_files)

# if st.button("Initialize Index"):
index_name = os.path.splitext(os.path.basename(selected_pdf))[0]

save_index_name(index_name)
index_names = load_index_names()

embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
texts = load_and_split_data(selected_pdf)

# pinecone.deinitialize()
pinecone.init(
    api_key="ef6b0907-1e0f-4b7e-a99d-b893c5686680",
    environment="eu-west1-gcp"
)
index = pinecone.Index(index_name)

# data = []
# for id,embedding in enumerate(embeddings):
#     metadata = {'metadata1': metadata_value}
#     data.append((id, embedding, metadata))
# index.upsert(data, namespace=index_name)

llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
chain = load_qa_chain(llm, chain_type="stuff")

query = st.text_input("Input question")
if query:
    docs = index.similarity_search(query, include_metadata=True, namespace=index_name)

    st.write(chain.run(input_documents=docs, question=query))

# else:
#     st.write("No PDF files found in the 'content' directory.")
