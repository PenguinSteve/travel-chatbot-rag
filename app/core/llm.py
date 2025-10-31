
from app.config.settings import settings
from langchain_groq import ChatGroq

def llm_plan() -> ChatGroq:
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = settings.LLM_MODEL
  
    temperature = settings.LLM_TEMPERATURE

    max_tokens = settings.LLM_MAX_TOKENS

    timeout = settings.LLM_TIMEOUT

    return ChatGroq(
        groq_api_key=api_key,
        model=model,    
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

def llm_summary() -> ChatGroq:
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("LLM_MODEL_SUMMARY not set")

    model = settings.LLM_MODEL_SUMMARY
    temperature = 0.0

    max_tokens = 1000

    timeout = settings.LLM_TIMEOUT

    return ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

def llm_chat() -> ChatGroq:
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("LLM_MODEL_CHAT not set")

    model = settings.LLM_MODEL_CHAT
    temperature = 0.0

    max_tokens = 1000

    timeout = settings.LLM_TIMEOUT

    return ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )