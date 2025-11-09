from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.controller import controller
from app.config.vector_database_pinecone import PineconeConfig
from app.repositories.pinecone_repository import PineconeRepository
from app.config.mongodb import get_database, get_docstore
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ParentDocumentRetriever

import os
from dotenv import load_dotenv

from app.services.reranker_service import RerankerService
load_dotenv()



@asynccontextmanager
async def life_span(app: FastAPI):
    try:

        # Initalize MongoDB connection
        db = get_database()
        app.state.db = db
        print('\n---------------------Connected to MongoDB database---------------------\n', db.name)
        
        # Initialize vector store
        vector_store = PineconeConfig().get_vector_store()

        # Initialize pinecone repository
        app.state.pinecone_repository = PineconeRepository(vector_store=vector_store)
        print('\n---------------------Initialized Pinecone repository with vector store---------------------\n')

        # Initialize Flashrank compressor
        app.state.flashrank_compressor = FlashrankRerank(top_n=3)
        print('\n---------------------Initialized Flashrank compressor---------------------\n')

        # Initialize Docstore mongodb
        docstore = get_docstore()
        app.state.docstore = docstore
        print('\n---------------------Initialized MongoDB docstore---------------------\n')

        # Initialize ParentDocumentRetriever
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""])

        app.state.parent_document_retriever = ParentDocumentRetriever(docstore=docstore,
                                                                    child_splitter=child_splitter, 
                                                                    vectorstore=vector_store,
                                                                    search_kwargs={"k":10, "filter":{} })
        print('\n---------------------Initialized ParentDocumentRetriever---------------------\n')

        # Initialize reranker service
        app.state.reranker_service = RerankerService()

    except Exception as e:
        raise RuntimeError(f"Failed to create vector_store/Pinecone repository/Flashrank compressor/Database connection at start up: {e}")

    yield

    # Shutdown
    print("\n---------------------Shutting down FastAPI application---------------------\n")
    app.state.pinecone_repository = None
    if hasattr(app.state, "db"):
        app.state.db.client.close()
        print("MongoDB connection closed")
    

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