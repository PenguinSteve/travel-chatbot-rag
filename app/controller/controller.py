from fastapi import APIRouter, Depends, HTTPException, Request
from pymongo import MongoClient
from app.models.chat_schema import ChatMessage
from app.repositories.pinecone_repository import PineconeRepository
from app.request.AskRequest import AskRequest
from app.response.AskResponse import AskResponse
from app.services.rag_service import RAGService
from app.core.dependencies import get_mongodb_instance, get_pinecone_repository, get_parent_document_retriever, get_flashrank_compressor
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from app.services.agent_service import AgentService
from app.repositories.chat_repository import ChatRepository
from app.utils.chat_history import build_chat_history_from_db
from langchain.retrievers import ParentDocumentRetriever

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, 
        pinecone_repository: PineconeRepository = Depends(get_pinecone_repository),
        flashrank_compressor: FlashrankRerank = Depends(get_flashrank_compressor),
        mongodb_instance: MongoClient = Depends(get_mongodb_instance),
        parent_document_retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever)
        ):

    print("\n---------------------Received Ask Request---------------------\n" \
    "Payload:", payload)
    
    message = payload.message
    session_id = payload.session_id

    # Validate input
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")

    chat_repository = ChatRepository(mongodb_instance)


    # Get all history messages from db
    past_messages = chat_repository.get_chat_history(session_id=session_id)

    chat_history = build_chat_history_from_db(past_messages)

    # Create standalone question from chat history
    standalone_question = RAGService.build_standalone_question(message, chat_history)

    print("\n---------------------Original question---------------------\n")
    print(message)

    print("\n---------------------Standalone question---------------------\n")
    print(standalone_question)

    # Classify query to get topic and location
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
        if 'Plan' in topics and len(locations) == 0:

            chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
            chat_repository.save_message(session_id=session_id, message=ChatMessage(content="Vui lòng cung cấp địa điểm để tôi có thể giúp bạn lập kế hoạch du lịch.", role="ai"))

            return AskResponse(message=payload.message, answer="Vui lòng cung cấp địa điểm để tôi có thể giúp bạn lập kế hoạch du lịch.")
        
        elif 'Plan' in topics:
            agent_service = AgentService(chat_repository, pinecone_repository, flashrank_compressor)
            response = agent_service.run_agent(question=standalone_question, session_id=session_id)
            response_text = response.get("output")
            return AskResponse(message=payload.message, answer=response_text)
        else :
            response_text, context_docs = RAGService.generate_response(parent_document_retriever, payload, standalone_question, chat_history, topics, locations, chat_repository)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution error: {e}")

    print("\n---------------------Context Documents:---------------------\n")
    for index, doc in enumerate(context_docs):
        if index > 0:
            print("--------------------------------------------------------------\n")
        print(f"Context number {index}:\n {doc.page_content}")
        print("  Metadata:", doc.metadata)
    print("\n---------------------End of Context Documents---------------------\n")

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