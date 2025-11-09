from app.services.rag_service import RAGService
from langchain.retrievers import ParentDocumentRetriever
from app.services.reranker_service import RerankerService
import json

def retrieve_document_rag_wrapper(tool_input: str, retriever: ParentDocumentRetriever = None, reranker: RerankerService = None): 
    payload = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
    topics = payload["topic"]    
    locations = payload["location"]
    query = payload["query"]

    return retrieve_document_rag(
        topics,
        locations,
        query,
        retriever,
        reranker
        )

def retrieve_document_rag(topics: list = [], locations: list = [], query: str = "", retriever: ParentDocumentRetriever = None, reranker: RerankerService = None):

    filter = {}
    if isinstance(topics, list) and len(topics) > 0:
                filter["Topic"] = {"$in": topics}
    if isinstance(locations, list) and len(locations) > 0:
        filter["Location"] = {"$in": locations}
        
    retriever.search_kwargs["filter"] = filter

    context_docs = RAGService.retrieve_documents(retriever=retriever, query=query, reranker=reranker)
    page_contents = [doc.page_content for doc in context_docs]
    return page_contents

