from llama_index import GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph

# do imports
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
import openai
import os
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, GraphToolConfig
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
openai.api_key = os.getenv("API_KEY")
from pathlib import Path 
import streamlit as st 

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
main_dir = Path("")
content_dir = Path("content")

json_files = list(main_dir.glob("*.json"))
multiselected_file = st.selectbox("Select an Index file:", json_files)


if multiselected_file:
    index = GPTSimpleVectorIndex.load_from_disk(os.path.join(main_dir, multiselected_file))
#     graph = ComposableGraph.from_indices(
#     GPTListIndex,
#     multiselected_file,
#     index_summaries=st.session_state.index_summaries,
# )

    decompose_transform = DecomposeQueryTransform(
        llm_predictor, verbose=True
    )

    # define query configs for graph 
    query_configs = [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs": {
                "similarity_top_k": 3,
                # "include_summary": True
            },
            "query_transform": decompose_transform
        }
    ]
    # graph config
    config = IndexToolConfig(
        index=index,
        name="My GPTSimpleVectorIndex",
        description="An index of some test data",
        index_query_kwargs={"similarity_top_k": 3},
        tool_kwargs={"return_direct": True}
    )

    toolkit = LlamaToolkit(
       index_configs=index_configs,
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm=OpenAI(temperature=0)
    agent_chain = create_llama_chat_agent(
        toolkit,
        llm,
        memory=memory,
        verbose=True
    )



    inp = st.text_input("Enter query")
    if inp:
        out = agent_chain.run(input=inp)
        st.write(out)