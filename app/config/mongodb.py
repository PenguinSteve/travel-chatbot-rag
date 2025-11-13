from pymongo import MongoClient
from langchain_community.storage import MongoDBStore
from app.config.settings import settings


def get_database() -> MongoClient:
    CONNECTION_STRING = f"mongodb+srv://{settings.MONGO_DB_NAME}:{settings.MONGO_DB_PASSWORD}@chat-box-tourism.ojhdj0o.mongodb.net/?retryWrites=true&w=majority&tls=true"
    client = MongoClient(CONNECTION_STRING)
    return client[settings.MONGO_DB_NAME]

def get_docstore() -> MongoDBStore:
    CONNECTION_STRING = f"mongodb+srv://{settings.MONGO_DB_NAME}:{settings.MONGO_DB_PASSWORD}@chat-box-tourism.ojhdj0o.mongodb.net/?retryWrites=true&w=majority&tls=true"

    docstore = MongoDBStore(
        connection_string=CONNECTION_STRING,
        db_name=settings.MONGO_DB_NAME,
        collection_name=settings.MONGO_STORE_COLLECTION_NAME
    )

    return docstore

def get_database_schedule() -> MongoClient:
    CONNECTION_STRING = f"mongodb+srv://{settings.MONGO_DB_NAME}:{settings.MONGO_DB_PASSWORD}@chat-box-tourism.ojhdj0o.mongodb.net/?retryWrites=true&w=majority&tls=true"
    client = MongoClient(CONNECTION_STRING)
    return client[settings.MONGO_DB_NAME_SCHEDULE]
