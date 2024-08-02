import os
from dotenv import load_dotenv
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

md_docs = FlatReader().load_data(Path("./data/fake_company.md"))

parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(md_docs)

client = QdrantClient(url="http://localhost:6333")


def create_index(nodes, colection_name):
    vector_store = QdrantVectorStore(colection_name, client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)


collection_name = "FakeCompany"

if __name__ == "__main__":
    create_index(nodes, collection_name)
