import json
from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_pinecone import PineconeRerank
from pymongo import MongoClient
from app.models.chat_schema import ChatMessage
from app.repositories.redis_chat_repository import RedisChatRepository
from app.request.AskRequest import AskRequest
from app.response.AskResponse import AskResponse
from app.services.rag_service import RAGService
from app.core.dependencies import (
    get_mongodb_instance,
    get_parent_document_retriever,
    get_pinecone_reranker,
    get_chat_repository)
from app.services.agent_service import AgentService
from app.repositories.chat_repository import ChatRepository
from app.utils.chat_history import build_chat_history_from_db
from langchain.retrievers import ParentDocumentRetriever
from app.middleware.auth_jwt import get_current_user_payload_strict

router = APIRouter()


@router.post("/create-schedule", response_model=AskResponse)
def create_schedule(payload: AskRequest,
        user_payload: dict = Depends(get_current_user_payload_strict),
        mongodb_instance: MongoClient = Depends(get_mongodb_instance),
        parent_document_retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
        pinecone_reranker: PineconeRerank = Depends(get_pinecone_reranker),
        ):
     
    print("\n---------------------Received Create Schedule Request---------------------\n"
    "Payload:", payload)
    message = payload.message
    session_id = payload.session_id

    user_id = user_payload.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: User ID missing.")

    # Validate input
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")
    
    chat_repository = ChatRepository(mongodb_instance, user_id)

    classify_result = RAGService.classify_query_for_schedule(message)
    topic = classify_result.get("Topic") or None
    location = classify_result.get("Location") or None

    if location is None:
        chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
        chat_repository.save_message(session_id=session_id, message=ChatMessage(content="Vui lòng cung cấp địa điểm để tôi có thể giúp bạn lập kế hoạch du lịch.", role="ai"))

        return AskResponse(message=payload.message, answer="Vui lòng cung cấp địa điểm để tôi có thể giúp bạn lập kế hoạch du lịch.")

    if topic is None or "Plan" not in topic:
        chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
        chat_repository.save_message(session_id=session_id, message=ChatMessage(content="Yêu cầu của bạn không liên quan đến việc lập kế hoạch du lịch. Vui lòng gửi yêu cầu khác.", role="ai"))

        return AskResponse(message=payload.message, answer="Yêu cầu của bạn không liên quan đến việc lập kế hoạch du lịch. Vui lòng gửi yêu cầu khác.")
    
    agent_service = AgentService(chat_repository=chat_repository, retriever=parent_document_retriever, pinecone_reranker=pinecone_reranker, user_id=user_id)
    try:
        response = agent_service.run_agent(question=message, session_id=session_id)
    except Exception as e:
        response = f"Tôi gặp lỗi trong khi tạo kế hoạch du lịch cho bạn, hãy thử lại sau."

    return AskResponse(message=payload.message, answer=response)

@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest,
        chat_repository: (ChatRepository | RedisChatRepository) = Depends(get_chat_repository),
        parent_document_retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
        pinecone_reranker: PineconeRerank = Depends(get_pinecone_reranker),
        ):

    print("\n---------------------Received Ask Request---------------------\n" \
    "Payload:", payload)
    
    message = payload.message
    session_id = payload.session_id

    # Validate input
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")

    # Get all history messages from db
    past_messages = chat_repository.get_chat_history(session_id=session_id)

    chat_history = build_chat_history_from_db(past_messages)

    print("\n---------------------Original question---------------------\n")
    print(message)
    standalone_question = None
    if(len(chat_history) > 0):
        # Create standalone question from chat history
        standalone_question = RAGService.build_standalone_question(message, chat_history).get("standalone_question", message)
        
        print("\n---------------------Standalone question---------------------\n")
        print(standalone_question)

    if(standalone_question is None):
        classify_result = RAGService.classify_query(message)
        standalone_question = message
    else:
        classify_result = RAGService.classify_query(standalone_question)

    topics = classify_result.get("Topic") or []
    locations = classify_result.get("Location") or []

    filter = {}
    if isinstance(topics, list) and len(topics) > 0:
        filter["Topic"] = {"$in": topics}
    if isinstance(locations, list) and len(locations) > 0:
        filter["Location"] = {"$in": locations}

    # Get retriever from Pinecone repository
    parent_document_retriever.search_kwargs["filter"] = filter

    # Generate response using RAG service
    try:
        response_text, context_docs = RAGService.generate_response(parent_document_retriever, payload, standalone_question, chat_history, topics, locations, chat_repository, pinecone_reranker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution error: {e}")

    return AskResponse(message=payload.message, answer=response_text)


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/health/db")
def check_database(request: Request):
    print("\n---------------------Checking Database Connection---------------------\n")
    try:
        db = request.app.state.db 
        db.command("ping")       
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")