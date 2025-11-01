from pymongo import MongoClient
from app.config.settings import settings
def get_database() -> MongoClient:
    CONNECTION_STRING = f"mongodb+srv://{settings.MONGO_DB_NAME}:{settings.MONGO_DB_PASSWORD}@chat-box-tourism.ojhdj0o.mongodb.net/?retryWrites=true&w=majority&tls=true"
    client = MongoClient(CONNECTION_STRING)
    return client[settings.MONGO_DB_NAME]
