from fastapi import Request
from app.repositories.pinecone_repository import PineconeRepository
from langchain_community.document_compressors import FlashrankRerank

# ✅ Dùng cả trong route và trong tool
def get_pinecone_repository(request: Request = None):
    try:
        # Nếu chạy trong FastAPI route → có request.app.state
        if request and hasattr(request.app.state, "pinecone_repository"):
            return request.app.state.pinecone_repository
    except Exception:
        pass

    # Nếu gọi từ LangChain tool (không có request)
    from app.config.vector_database_pinecone import PineconeConfig
    vector_store = PineconeConfig().get_vector_store()
    return PineconeRepository(vector_store=vector_store)


def get_flashrank_compressor(request: Request = None):
    try:
        if request and hasattr(request.app.state, "flashrank_compressor"):
            return request.app.state.flashrank_compressor
    except Exception:
        pass

    # Fallback: tạo thủ công khi chạy trong tool
    return FlashrankRerank(top_n=5)
