from fastapi import Request
from app.repositories.pinecone_repository import PineconeRepository

def get_pinecone_repository(request: Request) -> PineconeRepository:
    if not hasattr(request.app.state, 'pinecone_repository'):
        raise RuntimeError("Pinecone repository not initialized in app state.")
        
    return request.app.state.pinecone_repository