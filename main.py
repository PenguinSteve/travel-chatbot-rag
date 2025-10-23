from fastapi import APIRouter, FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.controller import controller
from app.config.vector_database_pinecone import PineconeConfig
from app.repositories.pinecone_repository import PineconeRepository
import os


@asynccontextmanager
async def life_span(app: FastAPI):
    try:
        # Initialize vector store
        vector_store = PineconeConfig().get_vector_store()

        # Initialize pinecone repository
        app.state.pinecone_repository = PineconeRepository(vector_store=vector_store)
    except Exception as e:
        raise RuntimeError(f"Failed to create vector_store/Pinecone repository at start up: {e}")

    yield

    # Shutdown
    print("\n---------------------Shutting down FastAPI application---------------------\n")
    app.state.pinecone_repository = None

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