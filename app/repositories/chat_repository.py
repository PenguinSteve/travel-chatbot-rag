from pymongo.collection import Collection
from datetime import datetime
from app.models.chat_schema import ChatMessage
from pymongo import MongoClient
class ChatRepository:
    def __init__(self, db: MongoClient):
        self.collection: Collection = db["chat_sessions"]
        
    def save_message(self, session_id: str, message: ChatMessage):
        self.collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message.model_dump()},
                "$set": {"updated_at": datetime.now()},
                "$setOnInsert": {"created_at": datetime.now()}
            },
            upsert=True
        )
        
    def get_chat_history(self, session_id: str):

        if session_id == "default":
            return []

        try:
            session = self.collection.find_one({"session_id": session_id})

            if session and "messages" in session:
                # Lấy 5 phần tử cuối cùng
                return session["messages"][-5:]
            else:
                return []

        except Exception as e:
            return []


    
    def get_all_sessions(self):
        return list(self.collection.find({}))
    