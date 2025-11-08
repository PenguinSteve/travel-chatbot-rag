from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    EMBEDDING_MODEL: str = "hiieu/halong_embedding"
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-tourism"
    DEFAULT_RAG_TOP_K: int = 5
    LLM_MODEL: str = "openai/gpt-oss-120b"
    PORT: int = 8080
    CORS_ALLOW_ORIGINS: str
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1000
    LLM_TIMEOUT: int = 60
    OPEN_WEATHER_API_KEY: str
    WEATHER_API_KEY: str
    LLM_MODEL_SUMMARY: str
    # LLM_MODEL: str = "openai/gpt-oss-120b"

    # LLM MODELS FOR RAG SYSTEM
    LLM_MODEL_RAG: str = "openai/gpt-oss-120b"
    LLM_MODEL_CLASSIFY: str = "llama-3.1-8b-instant"
    LLM_MODEL_CREATE_STANDALONE_QUESTION: str = "llama-3.1-8b-instant"

    # LLM MODELS FOR EVALUATION
    LLM_MODEL_EVALUATE: str = "openai/gpt-oss-120b"
    GROQ_API_KEY_FAITHFULNESS: str
    GROQ_API_KEY_RELEVANCE: str
    GROQ_API_KEY_PRECISION: str
    GROQ_API_KEY_RECALL: str

    # DATABASE SETTINGS
    MONGO_DB_NAME: str
    MONGO_DB_PASSWORD: str 
    MONGO_STORE_COLLECTION_NAME: str = "documents"

    class Config:
        env_file = ".env"
        

settings = Settings()