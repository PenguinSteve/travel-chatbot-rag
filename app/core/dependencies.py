from fastapi import Request
from app.repositories.pinecone_repository import PineconeRepository
from langchain_community.document_compressors import FlashrankRerank
from pymongo import MongoClient

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

