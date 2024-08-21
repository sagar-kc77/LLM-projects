import sys
import logging
import html2text
import nest_asyncio
import streamlit as st
import yaml, os, openai, textwrap
from llama_index.llms import OpenAI
from requests_html import HTMLSession
from llama_index.llms import AzureOpenAI
from llama_index.llm_predictor import LLMPredictor
from llama_index import set_global_service_context
from typing import Any, Dict, Generator, List, Union
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import StorageContext, load_index_from_storage
from llama_index import ServiceContext, SimpleDirectoryReader, TreeIndex, VectorStoreIndex

with open('C:/Work Space/LLM RESEARCH/awesome-llm-projects/cadentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

chat_llm = LLMPredictor(
                        llm=AzureOpenAI(
                                        deployment_name=credentials['AZURE_DEPLOYMENT_NAME'],
                                        model=credentials['AZURE_ENGINE'],
                                        api_key=credentials['AZURE_OPENAI_KEY'],
                                        api_version=credentials['AZURE_OPENAI_VERSION'],
                                        azure_endpoint=credentials['AZURE_OPENAI_BASE']
                                        )
                        )
embedding_llm = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

service_context = ServiceContext.from_defaults(
                                                llm_predictor=chat_llm,
                                                embed_model=embedding_llm,
                                                )

set_global_service_context(service_context)

def load_index(knowledge_base_dir: str) -> VectorStoreIndex:
    """Load the vector index from the directory."""
    print("Loading vector index...")
    storage_context = StorageContext.from_defaults(persist_dir=knowledge_base_dir)
    index = load_index_from_storage(storage_context=storage_context)
    query_engine = index.as_query_engine()
    print("Done.")
    return query_engine

query_engine = load_index('kb/')

@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def load_index() -> Any:
    """Load the index from the storage directory."""
    print("Loading index...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(base_dir, "kb")

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=dir_path)
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    print("Done.")
    return query_engine
    

if __name__ == "__main__":
    main()
