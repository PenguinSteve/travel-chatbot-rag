from http.client import HTTPException
from fastapi import Request, Header
from fastapi.params import Depends
from redis import Redis
from app.core.security import decode_access_token
from app.middleware.auth_jwt import get_user_payload_optional
from app.repositories.chat_repository import ChatRepository
from app.repositories.pinecone_repository import PineconeRepository
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.storage import MongoDBStore
from pymongo import MongoClient
from langchain.retrievers import ParentDocumentRetriever
from app.config.redis_cache import get_redis_instance
from app.repositories.redis_chat_repository import RedisChatRepository

def get_pinecone_repository(request: Request) -> PineconeRepository:
    if not hasattr(request.app.state, 'pinecone_repository'):
        raise RuntimeError("Pinecone repository not initialized in app state.")
        
    return request.app.state.pinecone_repository

def get_flashrank_compressor(request: Request) -> FlashrankRerank:
    if not hasattr(request.app.state, 'flashrank_compressor'):
        raise RuntimeError("FlashRank compressor not initialized in app state.")
        
    return request.app.state.flashrank_compressor

def get_mongodb_instance(request: Request) -> MongoClient:
    if not hasattr(request.app.state, 'db'):
        raise RuntimeError("MongoDB instance not initialized in app state.")
        
    return request.app.state.db

def get_docstore(request: Request) -> MongoDBStore:
    if not hasattr(request.app.state, 'docstore'):
        raise RuntimeError("MongoDBStore not initialized in app state.")

    return request.app.state.docstore

def get_parent_document_retriever(request: Request) -> ParentDocumentRetriever:
    if not hasattr(request.app.state, 'parent_document_retriever'):
        raise RuntimeError("ParentDocumentRetriever not initialized in app state.")

    return request.app.state.parent_document_retriever

def get_reranker_service(request: Request):
    if not hasattr(request.app.state, 'reranker_service'):
        raise RuntimeError("RerankerService not initialized in app state.")

    return request.app.state.reranker_service

def get_pinecone_reranker(request: Request):
    if not hasattr(request.app.state, 'pinecone_reranker'):
        raise RuntimeError("PineconeRerank not initialized in app state.")

    return request.app.state.pinecone_reranker

def get_redis_instance(request: Request):
    try:
        return request.app.state.redis_instance
    except AttributeError:
        # Dự phòng nếu redis chưa được khởi tạo
        redis_conn = get_redis_instance()
        request.app.state.redis_instance = redis_conn
        return redis_conn

def get_chat_repository(
    # Các dependency cơ sở
    mongodb_instance: MongoClient = Depends(get_mongodb_instance),
    redis_instance: Redis = Depends(get_redis_instance),
    # Dependency mới: Lấy payload người dùng
    user_payload: dict | None = Depends(get_user_payload_optional)
):
    """
    Quyết định cung cấp repository nào (Mongo/Redis)
    dựa trên việc JWT có hợp lệ hay không.
    """
    # Nếu user_payload không phải là None (tức là JWT hợp lệ)
    if user_payload:
        user_id = user_payload.get("id")

        if not user_id:
            raise HTTPException(401, "Invalid token: User ID missing.")

        print(f"\n---------------------Using MongoChatRepository (User: {user_id})---------------------\n")
        # Trả về repository dùng MongoDB
        return ChatRepository(mongodb_instance, user_id)
    
    # Nếu user_payload là None (khách hoặc token không hợp lệ)
    print("\n---------------------Using RedisChatRepository (Guest User)---------------------\n")
    # Trả về repository dùng Redis
    return RedisChatRepository(redis_instance)

def get_schedule_repository(request: Request):
    if not hasattr(request.app.state, 'schedule_repository'):
        raise RuntimeError("Schedule repository not initialized in app state.")
        
    return request.app.state.schedule_repository