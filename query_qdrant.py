from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from qdrant_client import QdrantClient

import os
from dotenv import load_dotenv

load_dotenv()

### Loading ENV variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

### Defining constants
TOP_K = 3

### Other components
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
collection_name = "OptisolTechNew"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

Settings.llm = llm

# print(llm)


### Loading existing store index
vector_store = QdrantVectorStore(collection_name, client=client)
index = VectorStoreIndex.from_vector_store(vector_store)

# print(index)

### Index to chat engine
retriever = VectorIndexRetriever(index, similarity_top_k=TOP_K)
query_engine = RetrieverQueryEngine(retriever=retriever)


def show_rag_nodes(source_nodes):
    for node in source_nodes:
        print(node.text)


def get_rag_response(query_response):
    return query_response.response


if __name__ == "__main__":
    questions = [
        "Describe the specific initiatives OptiSol Technologies has implemented as part of their Green IT Initiative.",
        "Who is the current Chief Marketing Officer (CMO) of OptiSol Technologies, and what is their background?",
        "What are the key components of OptiSol Technologies' incident response approach in the event of a cybersecurity breach?",
        "List and describe three major partnerships OptiSol Technologies has established for their cloud services.",
        "What are the primary focus areas for OptiSol Technologies' future R&D efforts, and how do these areas align with current industry trends?",
    ]
    for q in questions:
        response = query_engine.query(q)
        print(get_rag_response(response))
        # print("<--->Nodes:")
        # print(show_rag_nodes(response.source_nodes))
        print("===" * 15)
