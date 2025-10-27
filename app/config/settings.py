from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-tourism"
    DEFAULT_RAG_TOP_K: int = 5
    LLM_MODEL: str = "openai/gpt-oss-120b"
    PORT: int = 8080
    EMBEDDING_MODEL: str = "hiieu/halong_embedding"
    CORS_ALLOW_ORIGINS: str
    LLM_MODEL_SUMMARY: str 
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int | None = None
    LLM_TIMEOUT: int = 60
    OPEN_WEATHER_API_KEY: str
    WEATHER_API_KEY: str
    # LLM_MODEL: str = "openai/gpt-oss-120b"

    class Config:
        env_file = ".env"
        

settings = Settings()