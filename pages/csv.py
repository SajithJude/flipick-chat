from pathlib import Path
from llama_index import download_loader

PandasCSVReader = download_loader("PandasCSVReader")

loader = PandasCSVReader()

uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

if uploaded_file is not None:
    documents = loader.load_data(file=Path('./transactions.csv'))
    index = GPTSimpleVectorIndex.from_documents(documents)
    prompt = st.text_input("Enter you question")
    ans = index.query(prompt)
    st.write(ans)
