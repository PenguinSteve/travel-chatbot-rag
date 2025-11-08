from app.config.settings import settings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import os

class PineconeConfig:
    def __init__(self):
        self.pinecone_api_key = settings.PINECONE_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        self.dimension = 768
        self.embedding_model = settings.EMBEDDING_MODEL

    def get_pinecone_index(self):
        pc = Pinecone(api_key=self.pinecone_api_key)

        if self.index_name not in pc.list_indexes().names():
            print(f"\n---------------------Creating Pinecone index: '{self.index_name}'---------------------\n")
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"\n---------------------Created new Pinecone index: '{self.index_name}'---------------------\n")
        else:
            print(f"\n---------------------Found existing Pinecone index: '{self.index_name}'---------------------\n")

        return pc.Index(self.index_name)

    def get_vector_store(self) -> PineconeVectorStore:

        # Connect to Pinecone index
        time_start_connection = os.times()
        print(f"\n---------------------Connecting to Pinecone index: '{self.index_name}'---------------------\n")
        pinecone_index = self.get_pinecone_index()
        time_end_connection = os.times()
        print(f"\n---------------------Connected to Pinecone index: '{self.index_name}' in", time_end_connection.user - time_start_connection.user, "seconds---------------------\n")


        # Initialize embedding model and vector store
        time_start_vector_store = os.times()
        print(f"\n---------------------Initializing Pinecone vector store with embedding model: {self.embedding_model}---------------------\n")
        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model)
        vector_store = PineconeVectorStore(index=pinecone_index, embedding=embedding_model)
        time_end_vector_store = os.times()
        print(f"\n---------------------Initialized Pinecone vector store in", time_end_vector_store.user - time_start_vector_store.user, "seconds---------------------\n")

        print("\n---------------------Connected to Pinecone vector store----------------------\n")
        return vector_store
