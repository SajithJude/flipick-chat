import os
from pathlib import Path 
from llama_index import download_loader, GPTSimpleVectorIndex, ServiceContext
import streamlit as st
import openai 

openai.api_key = os.getenv("API_KEY")


UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()
service_context = ServiceContext.from_defaults(chunk_size_limit=512)
index_set = {}
content_dir = Path("content")
for pdf_file in content_dir.glob("*.pdf"):
    file_path = str(pdf_file.resolve())
    st.write(file_path)
    year_docs = loader.load_data(file=Path(file_path), split_documents=False)
    cur_index = GPTSimpleVectorIndex.from_documents(year_docs, service_context=service_context)
    index_set[file_name] = cur_index
    cur_index.save_to_disk(f'index_{file_name}.json')
    st.success("index saved for "+ file_name)

# st.write(len(content_dir.glob("*.json")))

# index_set = {}

# content_dir = Path("content")
# for pdf_file in content_dir.glob("*.pdf"):
#     file_path = str(pdf_file.resolve())
#     file_name = pdf_file.stem
#     cur_index = GPTSimpleVectorIndex.from_documents({file_name: file_path}, service_context=service_context)
#     index_set[file_name] = cur_index
#     cur_index.save_to_disk(f'index_{file_name}.json')
#     st.write(f'index_{file_name}.json')
