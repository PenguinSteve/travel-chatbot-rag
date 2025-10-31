from fastapi import APIRouter, Depends, HTTPException, Request
from pymongo import MongoClient
from app.models.chat_schema import ChatMessage
from app.repositories.pinecone_repository import PineconeRepository
from app.request.AskRequest import AskRequest
from app.response.AskResponse import AskResponse
from app.services.rag_service import RAGService
from app.core.dependencies import get_mongodb_instance, get_pinecone_repository
from app.core.dependencies import get_flashrank_compressor
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from app.services.agent_service import AgentService
from app.services.chat_history import chat_history_to_messages
from app.repositories.chat_repository import ChatRepository
from app.utils.chat_history import build_chat_history_from_db

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, 
        pinecone_repository: PineconeRepository = Depends(get_pinecone_repository),
        flashrank_compressor: FlashrankRerank = Depends(get_flashrank_compressor),
        mongodb_instance: MongoClient = Depends(get_mongodb_instance)):

    print("\n---------------------Received Ask Request---------------------\n" \
    "Payload:", payload)
    
    message = payload.message
    session_id = payload.session_id

    # Validate input
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")

    chat_repository = ChatRepository(mongodb_instance)

    print("Check repository")

    # Get all history messages from db
    past_messages = chat_repository.get_chat_history(session_id=session_id)


    print("Past messages from DB:", past_messages)

    chat_history = build_chat_history_from_db(past_messages)

    print("Chat history", chat_history)

    # Create standalone question from chat history
    standalone_question = RAGService.build_standalone_question(message, chat_history)

    print("Standalone question", standalone_question)

    # Classify query to get topic and location
    classify_result = RAGService.classify_query(standalone_question)
    print("\n---------------------Classify Result---------------------\n")
    print(classify_result)
    print("\n---------------------End of Classify Result---------------------\n")

    topic = classify_result.get("Topic") or None
    location = classify_result.get("Location") or None

    filter = {}
    if topic:
        filter["Topic"] = topic
    if location:
        filter["Location"] = location

    # Get retriever from Pinecone repository
    retriever = pinecone_repository.get_retriever(k=10, filter=filter)

    # Create compression retriever
    flashrank_compressor = flashrank_compressor
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=flashrank_compressor
    )

    # Generate response using RAG service
    try:
        if topic == 'Plan' & location == None:

            chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
            chat_repository.save_message(session_id=session_id, message=ChatMessage(content="Vui lòng cung cấp địa điểm để tôi có thể giúp bạn lập kế hoạch du lịch.", role="ai"))

            return AskResponse(message=payload.message, answer="Vui lòng cung cấp địa điểm để tôi có thể giúp bạn lập kế hoạch du lịch.")
        
        elif topic == 'Plan':
            agent_service = AgentService(pinecone_repository, flashrank_compressor)
            response = agent_service.run_agent(question=payload.message)
            response_text = response.get("output")
            return AskResponse(message=payload.message, answer=response_text)
        else :
            response_text, context_docs = RAGService.generate_groq_response(compression_retriever, payload, standalone_question, chat_history, topic, location, chat_repository)
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