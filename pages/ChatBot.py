from llama_index import download_loader, GPTSimpleVectorIndex, ServiceContext
from pathlib import Path
import streamlit as st
years = [1,2]
UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)

loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(file=Path(f'content/*_{year}.pdf'), split_documents=False)
    st.write(year_docs)
    # insert year metadata into each year
    for d in year_docs:
        d.extra_info = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)
    # content\HAND BOOK ON DEBT RECOVERY_1.pdf