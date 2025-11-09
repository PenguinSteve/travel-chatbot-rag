from fastapi import Request
from app.repositories.pinecone_repository import PineconeRepository
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.storage import MongoDBStore
from pymongo import MongoClient
from langchain.retrievers import ParentDocumentRetriever

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
