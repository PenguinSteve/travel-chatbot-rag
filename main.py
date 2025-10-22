from fastapi import APIRouter, FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.api import controller
from app.services.rag_service import RAGService
from app.config.settings import settings
import os

# Global retriever (khởi tạo 1 lần)
RETRIEVER = None

@asynccontextmanager
async def life_span(app: FastAPI):
    global RETRIEVER

    INDEX_NAME = settings.PINECONE_INDEX_NAME
    DEFAULT_K = settings.RAG_TOP_K

    try:
        RETRIEVER = RAGService.get_retriever(index_name=INDEX_NAME, k=DEFAULT_K)
    except Exception as e:
        raise RuntimeError(f"Failed to create retriever at startup: {e}")
    yield

    # Shutdown
    RETRIEVER = None

app = FastAPI(lifespan=life_span)

router = APIRouter()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(controller.router)

def get_retriever():
    return RETRIEVER

controller.set_retriever_getter(get_retriever)