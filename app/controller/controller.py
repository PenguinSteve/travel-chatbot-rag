from fastapi import APIRouter, Depends, HTTPException, Request
from app.repositories.pinecone_repository import PineconeRepository
from app.request.AskRequest import AskRequest, ChatRequest
from app.response.AskResponse import AskResponse, ChatResponse
from app.services.rag_service import RAGService
from app.core.dependencies import get_pinecone_repository
from app.core.dependencies import get_flashrank_compressor
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from app.services.agent_service import AgentService
from app.services.chat_history import chat_history_to_messages
from app.repositories.chat_repository import ChatRepository
from app.models.chat_schema import ChatMessage 
from app.utils.chat_history import build_chat_history_from_db

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, 
        pinecone_repository: PineconeRepository = Depends(get_pinecone_repository),
        flashrank_compressor: FlashrankRerank = Depends(get_flashrank_compressor)):
    
    # Validate input
    if not payload.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")

    classify_result = RAGService.classify_query(payload.query)
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
        if topic == 'Plan':
            agent_service = AgentService(pinecone_repository, flashrank_compressor)
            response = agent_service.run_agent(question=payload.query)
            response_text = response.get("output")
            return AskResponse(query=payload.query, answer=response_text)
        else :
            response_text, context_docs = RAGService.generate_groq_response(compression_retriever, payload.query, topic, location)
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


@router.post("/chat")
def chat_history(request: Request, payload: ChatRequest, 
        pinecone_repository: PineconeRepository = Depends(get_pinecone_repository),
        flashrank_compressor: FlashrankRerank = Depends(get_flashrank_compressor)):
    print("---> Received chat request:", payload)
    db = request.app.state.db
    chat_repo = ChatRepository(db)
    session_id = payload.session_id
    message_text = payload.message
    
    # Get all history messages from db
    past_messages = chat_repo.get_chat_history(session_id=session_id)
    chat_history = build_chat_history_from_db(past_messages)
    
    chat_repo.save_message(session_id=session_id, message=ChatMessage(content=message_text, role="human"))
    
    classify_result = RAGService.classify_query(payload.message)

    topic = classify_result.get("Topic") or None
    location = classify_result.get("Location") or None

    filter = {}
    if topic:
        filter["Topic"] = topic
    if location:
        filter["Location"] = location

    retriever = pinecone_repository.get_retriever(k=10, filter=filter)
    

    response = chat_history_to_messages(retriever=retriever, question=payload.message, session_id=session_id, chat_repo=chat_repo, chat_history=chat_history)
    return ChatResponse(chat_history=response['message'], reformulated_question=response['docs'], final_answer=response['final_answer'])


@router.get("/health")
def health_check():
    return {"status": "ok"}