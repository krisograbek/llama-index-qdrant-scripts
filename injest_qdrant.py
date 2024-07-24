import os
from dotenv import load_dotenv
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

md_docs = FlatReader().load_data(Path("./data/optisol_info.md"))

parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(md_docs)


client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# print(client.get_collections())


def create_index(nodes, colection_name):
    vector_store = QdrantVectorStore(colection_name, client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)


collection_name = "OptisolTechNew"

if __name__ == "__main__":
    create_index(nodes, collection_name)
