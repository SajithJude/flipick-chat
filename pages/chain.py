from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import os 
import streamlit as st 


loader = UnstructuredPDFLoader("content/Treasury Management Book .pdf")
data = loader.load()
st.write(f'You have {len(data)} document(s) in your data')
st.write(f'There are {len(data[0].page_content)} characters in your document')



text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

st.write(f'Now you have {len(texts)} documents')

openai.api_key = os.getenv("API_KEY")



embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# initialize pinecone
pinecone.init(
    api_key="ef6b0907-1e0f-4b7e-a99d-b893c5686680",  # find at app.pinecone.io
    environment="eu-west1-gcp" # next to api key in console
)
index_name = "langchain-openai"
namespace = "book"

docsearch = Pinecone.from_texts(
  [t.page_content for t in texts], embeddings,
  index_name=index_name, namespace=namespace)


llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
chain = load_qa_chain(llm, chain_type="stuff")

query =st.text_input("Input question")
if query:
    docs = docsearch.similarity_search(query,
    include_metadata=True, namespace=namespace)

    st.write(chain.run(input_documents=docs, question=query))