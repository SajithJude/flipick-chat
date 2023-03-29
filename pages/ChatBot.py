import os
from pathlib import Path 
from llama_index import download_loader, GPTSimpleVectorIndex, ServiceContext
import streamlit as st

index_set = {}
service_context = ServiceContext.from_defaults(chunk_size_limit=512)

content_dir = Path("content")
for pdf_file in content_dir.glob("*.pdf"):
    file_path = str(pdf_file.resolve())
    file_name = pdf_file.stem
    cur_index = GPTSimpleVectorIndex.from_documents({file_name: file_path}, service_context=service_context)
    index_set[file_name] = cur_index
    cur_index.save_to_disk(f'index_{file_name}.json')
    st.write(f'index_{file_name}.json')
