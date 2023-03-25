import streamlit as st
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt
import os
import PyPDF2

# import streamlit as st
import openai 
# from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader,PromptHelper,QuestionAnswerPrompt
# import os 
from streamlit_chat import message as st_message

favicon = "favicon.ac8d93a.69085235180674d80d902fdc4b848d0b.png"
st.set_page_config(page_title="Flipick Chat", page_icon=favicon)

openai.api_key = os.getenv("API_KEY")

# Define function to extract text from PDF
def extract_text(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page).extractText()
    return text

# Define function to create and save the index
def create_index(pdf_files):
    # Load the PDF files and extract text
    texts = []
    for pdf_file in pdf_files:
        with open("content/"+pdf_file, 'rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()
            texts.append(text)

    # Create documents
    documents = [Document(texts[0], id=os.path.basename(pdf_files[0])),
                 Document(texts[1], id=os.path.basename(pdf_files[1]))]

    # Create and save the index
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk("index.json")

    return index

if "history" not in st.session_state:
    st.session_state.history = []

def generate_answer():
    user_message = st.session_state.input_text
    
    if any(op in user_message for op in ['+', '-', '*', '/', '%']):
        st.session_state.history.append({"message": user_message, "is_user": True})
        st.session_state.history.append({"message": "I'm sorry, I'm not allowed to perform calculations.", "is_user": False})
    else:
        query_str = str(user_message)
        context_str = "Generate answers for the questions that are relevant only to the documents context, throw a default answer saying I dont know for unrelevant questions."
        QA_PROMPT_TMPL = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query_str}\n"
        )
        QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
        message_bot = index.query(query_str, text_qa_template=QA_PROMPT, response_mode="compact", mode="embedding")
        # source = message_bot.get_formatted_sources()
        # st.sidebar.write("Answer Source :",source)  # added line to display source on sidebar
        st.session_state.history.append({"message": user_message, "is_user": True})
        st.session_state.history.append({"message": str(message_bot), "is_user": False})

# Create the app layout
# st.title("PDF Indexer")
with st.expander("Upload pdfs and create index"):
    pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    # If PDF files are uploaded, create and save the index
    if pdf_files:
        directory_path = "content/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        for pdf_file in pdf_files:
            with open(os.path.join(directory_path, pdf_file.name), "wb") as f:
                f.write(pdf_file.getbuffer())
        # index_file_name = "Combined_index.json"
        # index = create_index(directory_path, index_file_name)
        # st.write(index_file_name)
        st.success(f"PDF succesfully uploaded to Path {directory_path}")

    # List all json
     # List all json files in the directory
    files = os.listdir('.')
    psdfs = os.listdir('content/')
    json_files = [f for f in files if f.endswith('.json')]
    pdf_files = [f for f in psdfs if f.endswith('.pdf')]

    # Create a dropdown to select the index file
    # index_file = st.selectbox("Select an index file:", json_files)
    document_files_pdf = st.multiselect("LIst of available PDFs:", pdf_files)
    if document_files_pdf:
        index = create_index(document_files_pdf)
        st.success(f"Index successfully created with {str(document_files_pdf)}")

   

col1, col2 = st.columns([1.4, 1])
col2.image("Flipick_Logo-1.jpg", width=210)
st.write("")
st.write("")

input_text = st.text_input("Ask flipick bot a question", key="input_text", on_change=generate_answer)
st.caption("Disclaimer : This ChatBOT is a pilot built solely for the purpose of a demo to Indian Institute of Banking and Finance (IIBF). The BOT has been trained based on the book Treasury Management published by IIBF. All content rights vest with IIBF")

# Display the conversation history
for chat in st.session_state.history:
    st_message(**chat)
