from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.controller import admin_controller, controller
from app.config.vector_database_pinecone import PineconeConfig
from app.repositories.pinecone_repository import PineconeRepository
from app.config.mongodb import get_database, get_docstore, get_database_schedule
from langchain.retrievers import ParentDocumentRetriever
from langchain_pinecone import PineconeRerank
from app.config.settings import settings
from app.config.redis_cache import get_redis_instance

import os
from dotenv import load_dotenv

from app.repositories.schedule_repository import ScheduleRepository

# from app.services.reranker_service import RerankerService
load_dotenv()



@asynccontextmanager
async def life_span(app: FastAPI):
    try:

        # Initalize MongoDB connection
        db = get_database()
        app.state.db = db
        print('\n---------------------Connected to MongoDB database---------------------\n', db.name)

        # Initialize schedule repository
        db_schedule = get_database_schedule()
        schedule_repo = ScheduleRepository(db_schedule)
        app.state.schedule_repository = schedule_repo
        print('\n---------------------Initialized Schedule repository---------------------\n')

        # Initialize vector store
        vector_store = PineconeConfig().get_vector_store()

        # Initialize pinecone repository
        app.state.pinecone_repository = PineconeRepository(vector_store=vector_store)
        print('\n---------------------Initialized Pinecone repository with vector store---------------------\n')

        # Initialize Docstore mongodb
        docstore = get_docstore()
        app.state.docstore = docstore
        print('\n---------------------Initialized MongoDB docstore---------------------\n')

        # Initialize ParentDocumentRetriever 

        # Child splitter
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""])
        
        # Parent splitter
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        app.state.parent_document_retriever = ParentDocumentRetriever(docstore=docstore,
                                                                    child_splitter=child_splitter,
                                                                    parent_splitter=parent_splitter,
                                                                    vectorstore=vector_store,
                                                                    search_kwargs={"k":15, "filter":{} })
        print('\n---------------------Initialized ParentDocumentRetriever---------------------\n')

        # # Initialize reranker service
        # app.state.reranker_service = RerankerService()

        # Initialize pinecone reranker
        app.state.pinecone_reranker = PineconeRerank(pinecone_api_key=settings.PINECONE_API_KEY_RERANKER, top_n=3)
        print('\n---------------------Initialized PineconeRerank---------------------\n')

        # Initialize redis cache
        app.state.redis_instance = get_redis_instance()
        print('\n---------------------Initialized Redis cache instance---------------------\n')

    except Exception as e:
        raise RuntimeError(f"Failed to create vector_store/Pinecone repository/Flashrank compressor/Database connection at start up: {e}")

    yield

    # Shutdown
    print("\n---------------------Shutting down FastAPI application---------------------\n")
    app.state.pinecone_repository = None
    if hasattr(app.state, "db"):
        app.state.db.client.close()
        print("MongoDB connection closed")

    if hasattr(app.state, "redis_instance"):
        app.state.redis_instance.close()
        print("Redis connection closed")
    

app = FastAPI(lifespan=life_span)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(controller.router)
app.include_router(admin_controller.router)