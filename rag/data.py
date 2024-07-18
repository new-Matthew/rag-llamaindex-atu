from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
import os
import argparse
import yaml
import qdrant_client
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding

from llama_index import ServiceContext
from llama_index.llms import Ollama


class Data:
    def __init__(self, config):
        self.config = config

    def _verify_data_folder(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data folder {data_path} does not exist.")
        if not any(file.endswith(".pdf") for file in os.listdir(data_path)):
            raise FileNotFoundError(f"No PDF files found in {data_path}.")
        print("Data folder verified and contains PDF files.")

    def ingest(self, embedder, llm):
        print("Indexing data...")
        self._verify_data_folder(self.config["data_path"])
        documents = SimpleDirectoryReader(self.config["data_path"]).load_data()

        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config["collection_name"]
        )
        storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embedder, chunk_size=self.config["chunk_size"]
        )

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, service_context=service_context
        )
        print(
            f"Data indexed successfully to Qdrant. Collection: {self.config['collection_name']}"
        )
        return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--ingest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ingest data to Qdrant vector Database.",
    )

    args = parser.parse_args()
    config_file = "config.yml"
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)
    data = Data(config)
    if args.ingest:
        print("Loading Embedder...")
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=config["embedding_model"])
        )
        llm = Ollama(model=config["llm_name"], base_url=config["llm_url"])
        data.ingest(embedder=embed_model, llm=llm)
