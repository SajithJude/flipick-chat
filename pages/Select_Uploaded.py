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

def load_and_split_data(pdf_file):
    loader = UnstructuredPDFLoader(pdf_file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts

openai.api_key = os.getenv("API_KEY")

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def prepare_embeddings_and_pinecone_index(index_name):
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    pinecone.deinit()
    pinecone.init(
        api_key="ef6b0907-1e0f-4b7e-a99d-b893c5686680",
        environment="eu-west1-gcp"
    )
    docsearch = Pinecone(namespace=index_name)
    return embeddings, docsearch

index_name = "multipdf"
embeddings, docsearch = prepare_embeddings_and_pinecone_index(index_name)

llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
chain = load_qa_chain(llm, chain_type="stuff")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if 'pdf_texts' not in st.session_state:
    st.session_state.pdf_texts = {}

for uploaded_file in uploaded_files:
    if uploaded_file.name not in st.session_state.pdf_texts:
        texts = load_and_split_data(uploaded_file)
        st.session_state.pdf_texts[uploaded_file.name] = texts
        docsearch.upsert_vectors(
            {f"{uploaded_file.name}-{i}": embeddings.embed_text(t.page_content) for i, t in enumerate(texts)}
        )

book_names = list(st.session_state.pdf_texts.keys())
if book_names:
    selected_book = st.selectbox("Select a book to ask questions", book_names)

    query = st.text_input("Input question")
    if query:
        docs = docsearch.similarity_search(query, include_metadata=True, namespace=selected_book)

        st.write(chain.run(input_documents=docs, question=query))
else:
    st.write("No PDF files uploaded yet.")
