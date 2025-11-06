
from app.config.settings import settings
from langchain_groq import ChatGroq

def llm_plan() -> ChatGroq:
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = settings.LLM_MODEL
  
    temperature = settings.LLM_TEMPERATURE

    # max_tokens = settings.LLM_MAX_TOKENS

    timeout = settings.LLM_TIMEOUT

    return ChatGroq(
        groq_api_key=api_key,
        model=model,    
        temperature=temperature,
        # max_tokens=max_tokens,
        timeout=timeout,
    )

def llm_summary() -> ChatGroq:
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("LLM_MODEL_SUMMARY not set")

    model = settings.LLM_MODEL_SUMMARY
    temperature = 0.0

    # max_tokens = 1000

    timeout = settings.LLM_TIMEOUT

    return ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=temperature,
        # max_tokens=max_tokens,
        timeout=timeout,
    )

def llm_create_standalone_question() -> ChatGroq:
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = settings.LLM_MODEL_CREATE_STANDALONE_QUESTION
    temperature = 0.0

    # max_tokens = 1000

    timeout = settings.LLM_TIMEOUT

    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temperature,
        # max_tokens=max_tokens,
        timeout=timeout,
    )

def llm_rag() -> ChatGroq:
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    
    model = settings.LLM_MODEL_RAG
    temperature = 0.0

    return ChatGroq(
        model=model, 
        temperature=temperature, 
        api_key=api_key
    )

def llm_classify() -> ChatGroq:
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = settings.LLM_MODEL_CLASSIFY
    temperature = 0.0

    return ChatGroq(
        model=model, 
        temperature=temperature, 
        api_key=api_key
    )

def llm_evaluate_faithfulness() -> ChatGroq:
    api_key = settings.GROQ_API_KEY_FAITHFULNESS
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = settings.LLM_MODEL_EVALUATE
    temperature = 0.0

    return ChatGroq(
        model=model, 
        temperature=temperature, 
        api_key=api_key
    )

def llm_evaluate_relevance() -> ChatGroq:
    api_key = settings.GROQ_API_KEY_RELEVANCE
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = settings.LLM_MODEL_EVALUATE
    temperature = 0.0

    return ChatGroq(
        model=model, 
        temperature=temperature, 
        api_key=api_key
    )

def llm_evaluate_precision() -> ChatGroq:
    api_key = settings.GROQ_API_KEY_PRECISION
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = settings.LLM_MODEL_EVALUATE
    temperature = 0.0

    return ChatGroq(
        model=model, 
        temperature=temperature, 
        api_key=api_key
    )

def llm_evaluate_recall() -> ChatGroq:
    api_key = settings.GROQ_API_KEY_RECALL
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = settings.LLM_MODEL_EVALUATE
    temperature = 0.0

    return ChatGroq(
        model=model, 
        temperature=temperature, 
        api_key=api_key
    )