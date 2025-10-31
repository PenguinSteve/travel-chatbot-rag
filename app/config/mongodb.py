from pymongo import MongoClient
from app.config.settings import settings
def get_database():
    CONNECTION_STRING = f"mongodb+srv://{settings.MONGO_DB_NAME}:{settings.MONGO_DB_PASSWORD}@chat-box-tourism.ojhdj0o.mongodb.net/"
    client = MongoClient(CONNECTION_STRING)
    return client[settings.MONGO_DB_NAME]

if __name__ == "__main__":
    db = get_database()
    print("Connected to MongoDB database:", db.name)