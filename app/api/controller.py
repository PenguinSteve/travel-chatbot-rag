from fastapi import APIRouter, HTTPException
from app.request.AskRequest import AskRequest
from app.response.AskResponse import AskResponse
from app.services.rag_service import RAGService
from typing import Callable

router = APIRouter()

# Global getter function (sẽ được set từ main.py)
_retriever_getter: Callable = None

def set_retriever_getter(getter: Callable):
    global _retriever_getter
    _retriever_getter = getter

def get_retriever_dependency():
    if _retriever_getter is None:
        raise HTTPException(status_code=500, detail="Retriever getter not initialized")
    retriever = _retriever_getter()
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    return retriever

@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    retriever = get_retriever_dependency()
    if not payload.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")

    try:
        response_text, context_docs = RAGService.generate_groq_response(retriever, payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution error: {e}")

    print("\n---------------------Context Documents:---------------------\n")
    for index, doc in enumerate(context_docs):
        if index > 0:
            print("--------------------------------------------------------------\n")
        print(f"Context number {index}:\n {doc.page_content}")
        print("  Metadata:", doc.metadata)
    print("\n---------------------End of Context Documents---------------------\n")

    return AskResponse(query=payload.query, answer=response_text)

@router.get("/health")
def health_check():
    return {"status": "ok"}