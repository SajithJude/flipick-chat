import os
from pathlib import Path 
from llama_index import download_loader, GPTSimpleVectorIndex, ServiceContext
import streamlit as st
import openai 

from llama_index import GPTListIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph



# do imports
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, GraphToolConfig

openai.api_key = os.getenv("API_KEY")


UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()
service_context = ServiceContext.from_defaults(chunk_size_limit=512)
index_set = {}
content_dir = Path("content")
for pdf_file in content_dir.glob("*.pdf"):
    file_path = str(pdf_file.resolve())
    file_name = pdf_file.stem
    st.write(file_path)
    year_docs = loader.load_data(file=Path(file_path), split_documents=False)
    cur_index = GPTSimpleVectorIndex.from_documents(year_docs, service_context=service_context)
    index_set[file_name] = cur_index
    cur_index.save_to_disk(f'index_{file_name}.json')
    st.success("index saved for "+ file_name)




index_summaries = [f"{pdf_file.stem}" for pdf_file in content_dir.glob("*.pdf")]

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index_set[file.stem] for file in content_dir.glob("*.pdf")], 
    index_summaries=index_summaries,
    service_context=service_context,
)


graph.save_to_disk('allIndexGraph.json')
# [optional] load from disk, so you don't need to build graph from scratch
graph = ComposableGraph.load_from_disk(
    'allIndexGraph.json', 
    service_context=service_context,
)


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
