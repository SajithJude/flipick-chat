import os
from pathlib import Path 
from llama_index import download_loader, GPTSimpleVectorIndex, ServiceContext
import streamlit as st
import openai 

from llama_index import GPTListIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph

from llama_index.indices.query.query_transform.base import DecomposeQueryTransform


# do imports
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, GraphToolConfig

openai.api_key = os.getenv("API_KEY")


import shutil

# Add this function to save the uploaded file to the 'content' folder
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    file_name = uploaded_file.name
    save_path = f"content/{file_name}"
    save_uploaded_file(uploaded_file, save_path)
    st.success(f"File '{file_name}' saved to {save_path}")


UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()
service_context = ServiceContext.from_defaults(chunk_size_limit=512)
index_set = {}

content_dir = Path("content")
pdf_files = list(content_dir.glob("*.pdf"))

# Display the multi-select input for users to select PDF files
selected_files = st.multiselect("Select PDF files to create an index", options=[pdf_file.stem for pdf_file in pdf_files])


for pdf_file in pdf_files:
    if pdf_file.stem in selected_files:
        file_path = str(pdf_file.resolve())
        file_name = pdf_file.stem
        st.write(file_path)
        year_docs = loader.load_data(file=Path(file_path), split_documents=False)
        cur_index = GPTSimpleVectorIndex.from_documents(year_docs, service_context=service_context)
        index_set[file_name] = cur_index
        cur_index.save_to_disk(f'index_{file_name}.json')
        st.success("index saved for "+ file_name)




index_summaries  = selected_files

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

st.write(graph)
# define a decompose transform
decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)

# define query configs for graph 
query_configs = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 1,
            # "include_summary": True
        },
        "query_transform": decompose_transform
    },
    {
        "index_struct_type": "list",
        "query_mode": "default",
        "query_kwargs": {
            "response_mode": "tree_summarize",
            "verbose": True
        }
    },
]
# graph config
graph_config = GraphToolConfig(
    graph=graph,
    name=f"Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber.",
    query_configs=query_configs,
    tool_kwargs={"return_direct": True}
)

# define toolkit
index_configs = []
for pdf_file in content_dir.glob("*.pdf"):
    file_name = pdf_file.stem
    tool_config = IndexToolConfig(
        index=index_set[file_name], 
        name=f"Vector Index for {file_name}",
        description=f"useful for when you want to answer queries about the {file_name} PDF file",
        index_query_kwargs={"similarity_top_k": 3},
        tool_kwargs={"return_direct": True}
    )
    # index_configs.append(tool_config)

index_configs.append(tool_config)

if "index_configs" not in st.session_state:
    st.session_state.index_configs = index_configs
# toolkit = LlamaToolkit(
#       index_configs=index_configs,
#     graph_configs=[graph_config]
# )

# memory = ConversationBufferMemory(memory_key="chat_history")
# llm=OpenAI(temperature=0)
# agent_chain = create_llama_chat_agent(
#     toolkit,
#     llm,
#     memory=memory,
#     verbose=True
# )
# inp = st.text_input("Enter query")
# if inp:
#     out = agent_chain.run(input=inp)
#     st.write(out)