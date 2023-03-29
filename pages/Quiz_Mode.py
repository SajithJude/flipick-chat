import pickle
import io

import streamlit as st
# from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.prompts import PromptTemplate
from pathlib import Path
import os
import openai
from PyPDF2 import PdfFileReader

# load_dotenv()
OPENAI_KEY = os.getenv('API_KEY')

# Define the prompt templates for the chatbot
_template = """ Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI version of the {name} .
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
Question: {question}
=========
{context}
=========
Answer:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context", "name"])


# Define a function to read a PDF file and return its text content
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    content= ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        content+= page.extract_text()
    return content
    
# Load the Streamlit app
st.title('PDF AI QuizBot ✨')
# st.subheader("Follow [@jamescodez](https://twitter.com/jamescodez) on twitter for more!")

# Load the PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    content = read_pdf(uploaded_file)

    # Split the text content into chunks
    text_splitter = CharacterTextSplitter()
    chunks = text_splitter.split_text(content)

    # Create embeddings from the text chunks
    vectorStorePkl = Path("vectorstore.pkl")
    vectorStore = None
    print("Regenerating search index vector store..")
    vectorStore = FAISS.from_texts(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_KEY))
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorStore, f)

    # Create the chatbot model
    qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0, openai_api_key=OPENAI_KEY),
                                    vectorstore=vectorStore, qa_prompt=QA_PROMPT)

    # Start the conversation with the chatbot
    chat_history = []
    youtuberName = st.text_input(label="Name", placeholder="Enter the name of the AI")
    userInput = st.text_input(label="Question", placeholder="Ask the AI a question!")
    if userInput != "":
        result = qa({"name": youtuberName, "question": userInput, "chat_history": chat_history}, return_only_outputs=True)
        st.subheader(result["answer"])
