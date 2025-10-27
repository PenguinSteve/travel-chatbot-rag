from app.repositories.pinecone_repository import PineconeRepository
from fastapi import Depends
from langchain_community.document_compressors import FlashrankRerank
from app.core.dependencies import get_flashrank_compressor, get_pinecone_repository
from langchain.retrievers import ContextualCompressionRetriever
from app.services.rag_service import RAGService
import json


def retrieve_document_rag_wrapper(tool_input: str): 
    payload = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
    topic = payload["topic"]    
    location = payload["location"]
    query = payload["query"]
    return retrieve_document_rag(topic, location, query)

def retrieve_document_rag(topic: str, location: str, query:str, 
            pinecone_repository: PineconeRepository = Depends(get_pinecone_repository),
            flashrank_compressor: FlashrankRerank = Depends(get_flashrank_compressor)):
    
    if not isinstance(pinecone_repository, PineconeRepository):
        pinecone_repository = get_pinecone_repository()
    if not isinstance(flashrank_compressor, FlashrankRerank):
        flashrank_compressor = get_flashrank_compressor()
        
    filter = {}
    if topic:
        filter["Topic"] = topic
    if location:
        filter["Location"] = location
        
    retriever = pinecone_repository.get_retriever(k=10, filter=filter)
    # Create compression retriever
    flashrank_compressor = flashrank_compressor
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=flashrank_compressor
    )
    
    context_docs = RAGService.retrieve_documents(compression_retriever, query)
    print("\n---------------------Context Documents:---------------------\n")
    print(context_docs)
    return context_docs

    