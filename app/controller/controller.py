from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from langchain_pinecone import PineconeRerank
from pymongo import MongoClient
from app.models.chat_schema import ChatMessage
from app.repositories.schedule_repository import ScheduleRepository
from app.repositories.redis_chat_repository import RedisChatRepository
from app.request.AskRequest import AskRequest
from app.response.AskResponse import AskResponse
from app.services.rag_service import RAGService
from app.core.dependencies import (
    get_mongodb_instance,
    get_parent_document_retriever,
    get_pinecone_reranker,
    get_chat_repository,
    get_schedule_repository)
from app.services.agent_service import AgentService
from app.repositories.chat_repository import ChatRepository
from app.utils.chat_history import build_chat_history_from_db
from langchain.retrievers import ParentDocumentRetriever
from app.middleware.auth_jwt import get_current_user_payload_strict

router = APIRouter()


@router.post("/create-schedule", response_model=AskResponse)
async def create_schedule(
        request: Request,
        payload: AskRequest,
        user_payload: dict = Depends(get_current_user_payload_strict),
        mongodb_instance: MongoClient = Depends(get_mongodb_instance),
        parent_document_retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
        pinecone_reranker: PineconeRerank = Depends(get_pinecone_reranker),
        schedule_repository: ScheduleRepository = Depends(get_schedule_repository)
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

    past_messages = chat_repository.get_chat_history(session_id=session_id)

    chat_history = build_chat_history_from_db(past_messages)

    print("\n---------------------Original question---------------------\n")
    print(message)
    standalone_question = None
    if(len(chat_history) > 0):
        # Create standalone question from chat history
        standalone_res = await run_in_threadpool(RAGService.build_standalone_question, message, chat_history)

        standalone_question = standalone_res.get("standalone_question", message)
        
        print("\n---------------------Standalone question---------------------\n")
        print(standalone_question)

    if(standalone_question is None):
        classify_result = await run_in_threadpool(RAGService.classify_query, message)
        standalone_question = message
    else:
        classify_result = await run_in_threadpool(RAGService.classify_query_for_schedule, standalone_question)

    topic = classify_result.get("Topic") or None
    location = classify_result.get("Location") or None

    if location is None:
        chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
        ai_message = chat_repository.save_message(session_id=session_id, message=ChatMessage(content="Vui lòng chọn một trong ba địa điểm hiện được hỗ trợ: Hà Nội, TP.HCM hoặc Đà Nẵng, để tôi có thể giúp bạn xây dựng lịch trình du lịch phù hợp.", role="ai"))

        return AskResponse(message=payload.message, answer="Vui lòng chọn một trong ba địa điểm hiện được hỗ trợ: Hà Nội, TP.HCM hoặc Đà Nẵng, để tôi có thể giúp bạn xây dựng lịch trình du lịch phù hợp.", timestamp=ai_message.timestamp)

    if topic is None or "Plan" not in topic:
        chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
        ai_message = chat_repository.save_message(session_id=session_id, message=ChatMessage(content="Yêu cầu của bạn không liên quan đến việc lập kế hoạch du lịch. Vui lòng gửi yêu cầu khác.", role="ai"))

        return AskResponse(message=payload.message, answer="Có vẻ như yêu cầu của bạn chưa liên quan đến việc lập kế hoạch du lịch. Bạn vui lòng gửi lại yêu cầu khác để tôi có thể hỗ trợ chính xác hơn nhé!", timestamp=ai_message.timestamp)
    
    agent_service = AgentService(chat_repository=chat_repository, retriever=parent_document_retriever, pinecone_reranker=pinecone_reranker, user_id=user_id)

    if await request.is_disconnected():
        print("Client disconnected before Agent start")
        return Response(status_code=204)

    try:
        agent_response = await run_in_threadpool(
            agent_service.run_agent,
            question=standalone_question,
            session_id=session_id)
    except Exception as e:
        print(f"Error in create_schedule: {e}")
        response = f"Xin lỗi bạn, quá trình tạo kế hoạch du lịch vừa gặp phải lỗi không mong muốn. Có thể hệ thống đang gặp sự cố tạm thời. Bạn vui lòng thử lại sau để tôi có thể tiếp tục hỗ trợ bạn một cách chính xác hơn nhé!"
        raise HTTPException(status_code=500, detail=response)

    trip_id = agent_response.get("trip_id", None)

    if await request.is_disconnected():
        print(f"Client disconnected after Agent execution. Trip ID generated: {trip_id}")

        if trip_id:
            await run_in_threadpool(schedule_repository.delete_schedule_by_trip_id, str(trip_id))
        return Response(status_code=204)

    try:
        decoded_answer = agent_response.get("answer", "")
        ai_message_text = agent_response.get("ai_message_for_history")

        chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
        saved_ai_message = chat_repository.save_message(session_id=session_id, message=ChatMessage(content=ai_message_text, role="ai", trip_id=trip_id))

    except Exception as e:
        print(f"Error saving chat history: {e}")
        raise HTTPException(status_code=500, detail="Error saving messages to chat history")

    return AskResponse(
        message=payload.message,
        answer=decoded_answer,
        timestamp=saved_ai_message.timestamp)

@router.post("/ask", response_model=AskResponse)
async def ask(
        request: Request,
        payload: AskRequest,
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

        standalone_res = await run_in_threadpool(RAGService.build_standalone_question, message, chat_history)

        standalone_question = standalone_res.get("standalone_question", message)
        
        print("\n---------------------Standalone question---------------------\n")
        print(standalone_question)

    if(standalone_question is None):
        classify_result = await run_in_threadpool(RAGService.classify_query, message)
        standalone_question = message
    else:
        classify_result = await run_in_threadpool(RAGService.classify_query, standalone_question)

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

    if await request.is_disconnected():
        print("Client disconnected")
        return Response(status_code=204)
    
    try:
        rag_result = await run_in_threadpool(
            RAGService.generate_response,
            parent_document_retriever,
            payload,
            standalone_question,
            chat_history,
            topics,
            locations,
            pinecone_reranker)
    except Exception as e:
        print(f"RAG generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    if await request.is_disconnected():
        print("Client disconnected after RAG generation, skip DB save")
        return Response(status_code=204)
    
    # Save messages to chat history
    try:
            
        chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
        ai_message_content = rag_result.get("response")
        ai_message = chat_repository.save_message(session_id=session_id, message=ChatMessage(content=ai_message_content, role="ai"))
    except Exception as e:
        print(f"Error saving messages to chat history: {e}")
        raise HTTPException(status_code=500, detail="Error saving messages to chat history")

    return AskResponse(
        message=payload.message,
        answer=rag_result.get("response"),
        timestamp=ai_message.timestamp
    )
     


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