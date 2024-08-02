from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from qdrant_client import QdrantClient

import os
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
client = QdrantClient(url="http://localhost:6333")
collection_name = "FakeCompany"

Settings.llm = llm


### Loading existing store index
vector_store = QdrantVectorStore(collection_name, client=client)
index = VectorStoreIndex.from_vector_store(vector_store)

### Index to chat engine
retriever = VectorIndexRetriever(index)
query_engine = RetrieverQueryEngine(retriever=retriever)


def show_rag_nodes(source_nodes):
    for node in source_nodes:
        print(node.text)


def get_rag_response(query_response):
    return query_response.response


if __name__ == "__main__":
    questions = [
        "What are the biggest milestones?",
        "What are the key technologies?",
    ]
    for q in questions:
        response = query_engine.query(q)
        print(get_rag_response(response))
        # print("<--->Nodes:")
        # print(show_rag_nodes(response.source_nodes))
        print("===" * 15)
