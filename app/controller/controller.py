from fastapi import APIRouter, Depends, HTTPException
from app.repositories.pinecone_repository import PineconeRepository
from app.request.AskRequest import AskRequest
from app.response.AskResponse import AskResponse
from app.services.rag_service import RAGService
from app.core.dependencies import get_pinecone_repository

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, 
        pinecone_repository: PineconeRepository = Depends(get_pinecone_repository)):

    classify_result = RAGService.classify_query(payload.query)
    print("\n---------------------Classify Result---------------------\n")
    print(classify_result)
    print("\n---------------------End of Classify Result---------------------\n")

    topic = classify_result.get("Topic") or "General"
    location = classify_result.get("Location") or None

    filter = {}
    if topic:
        filter["Topic"] = topic
    if location:
        filter["Location"] = location

    # Get retriever from Pinecone repository
    retriever = pinecone_repository.get_retriever(filter=filter)

    # Validate input
    if not payload.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")

    # Generate response using RAG service
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