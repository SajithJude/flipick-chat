from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

embeddings = OpenAIEmbeddings(openai_api_key="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")

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


llm = OpenAI(temperature=0, openai_api_key="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")
chain = load_qa_chain(llm, chain_type="stuff")

query = "What is this document about"
docs = docsearch.similarity_search(query,
  include_metadata=True, namespace=namespace)

outpt = chain.run(input_documents=docs, question=query)

st.write(outpt)