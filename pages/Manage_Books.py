import streamlit as st
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt
import os
import PyPDF2

import openai 
from streamlit_chat import message as st_message

def display_pdf(directory_path, pdf_file):
    with open(os.path.join(directory_path, pdf_file), "rb") as f:
        pdf_reader = PyPDF2.PdfReader (f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            with st.expander('Page'):
                st.write(page.extract_text())

def delete_pdf(directory_path, pdf_file):
    os.remove(os.path.join(directory_path, pdf_file))
    
pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
if pdf_files:
    # Process the PDF files and create separate indexes for each file
    indexes = []
    with st.spinner('Uploading files...'):
        directory_path = "content/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        for pdf_file in pdf_files:
            with open(os.path.join(directory_path, pdf_file.name), "wb") as f:
                f.write(pdf_file.getbuffer())
            # Create a document and index for the PDF file
            document = Document(f.read(), pdf_file.name)
            index = GPTSimpleVectorIndex([document])
            index.save_to_disk(f"{pdf_file.name}.index")
            indexes.append(index)

    st.success(f"PDF files successfully uploaded to path {directory_path}. Creating indexes...")
    st.success(f"{len(indexes)} indexes created successfully.")
    

# Define the directory path
directory_path = "content/"

# Get a list of files in the directory
files = os.listdir(directory_path)

colms = st.columns((4, 1, 1))

fields = ["Name", 'View', 'Delete' ]
for col, field_name in zip(colms, fields):
    # header
    col.subheader(field_name)

i=1
for  Name in files:
    i+=1
    col1, col2, col3 = st.columns((4 , 1, 1))
    # col1.write(x)  # index
    col1.caption(Name)  # email
    col2.button("View", key=Name, on_click=display_pdf, args=(directory_path, Name))  # unique ID
    # col4.button(user_table['Delete'][x])   # email status
    delete_status = fields[0]  # flexible type of button
    button_type = "Delete" if delete_status else "Gone"
    button_phold = col3.empty()  # create a placeholder
    do_action = button_phold.button(button_type, key=i, on_click=delete_pdf, args=(directory_path, Name))
    # if do_action:
    #         pass # do some action with a row's data
    #         button_phold.empty()  #  remove button
