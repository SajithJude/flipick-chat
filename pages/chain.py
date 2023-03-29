from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import openai
import os
import streamlit as st

# Load the data and split it into chunks
@st.cache
def load_and_split_data():
    loader = UnstructuredPDFLoader("content/Treasury Management Book .pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts

texts = load_and_split_data()

openai.api_key = os.getenv("API_KEY")

# Embeddings and Pinecone index
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def prepare_embeddings_and_pinecone_index():
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    pinecone.init(
        api_key="ef6b0907-1e0f-4b7e-a99d-b893c5686680",
        environment="eu-west1-gcp"
    )
    index_name = "langchain-openai"
    namespace = "book"
    docsearch = Pinecone.from_texts(
        [t.page_content for t in texts], embeddings,
        index_name=index_name, namespace=namespace
    )
    return embeddings, docsearch

embeddings, docsearch = prepare_embeddings_and_pinecone_index()

llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
chain = load_qa_chain(llm, chain_type="stuff")

index_name = "langchain-openai"
namespace = "book"

query = st.text_input("Input question")
if query:
    docs = docsearch.similarity_search(query,include_metadata=True, namespace=namespace)
    st.write(chain.run(input_documents=docs, question=query))
