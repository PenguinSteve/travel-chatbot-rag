import os
from typing import List
from langchain_community.storage import MongoDBStore
import pandas as pd
from app.config.settings import settings
from app.config.vector_database_pinecone import PineconeConfig
from app.services.data_service import DataService
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    CONNECTION_STRING = f"mongodb+srv://{settings.MONGO_DB_NAME}:{settings.MONGO_DB_PASSWORD}@chat-box-tourism.ojhdj0o.mongodb.net/?retryWrites=true&w=majority&tls=true"
    script_path = os.path.abspath(__file__)

    script_dir = os.path.dirname(script_path)

    project_root = os.path.dirname(script_dir)
    
    file_path = os.path.join(project_root, "data", "data_tourism_TPHCM.xlsx")
    
    docstore = MongoDBStore(
        connection_string=CONNECTION_STRING,
        db_name=settings.MONGO_DB_NAME,
        collection_name=settings.MONGO_STORE_COLLECTION_NAME
    )
    vector_store = PineconeConfig().get_vector_store()

    documents = DataService.load_raw_data(filepath=file_path)

    print(f"\n---------------------Loaded {len(documents)} raw documents---------------------\n")

    DataService.ingest_data(documents, docstore, vector_store)

if __name__ == "__main__":
    main()
    # pass