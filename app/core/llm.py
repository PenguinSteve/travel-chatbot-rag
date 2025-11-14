
from app.config.settings import settings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

def _gemini(model: str, temperature: float = 0.0, max_output_tokens: int = 4096) -> ChatGoogleGenerativeAI:
    api_key = settings.GEMINI_API_KEY  
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        google_api_key=api_key,
    )

def llm_plan() -> ChatGoogleGenerativeAI:
    # nhanh, rẻ cho planning / hành động agent
    return _gemini(model="gemini-2.5-flash-lite", temperature=settings.LLM_TEMPERATURE or 0.3)

# def llm_plan() -> ChatGroq:
#     api_key = settings.GROQ_API_KEY
#     if not api_key:
#         raise RuntimeError("GROQ_API_KEY not set")

#     model = settings.LLM_MODEL
  
#     temperature = settings.LLM_TEMPERATURE

#     # max_tokens = settings.LLM_MAX_TOKENS

#     timeout = settings.LLM_TIMEOUT

#     return ChatGroq(
#         groq_api_key=api_key,
#         model=model,    
#         temperature=temperature,
#         # max_tokens=max_tokens,
#         timeout=timeout,
#     )

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

def llm_create_standalone_question() -> ChatGoogleGenerativeAI:
    api_key = settings.GEMINI_API_KEY_CREATE_STANDALONE_QUESTION
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY_CREATE_STANDALONE_QUESTION not set")

    model = settings.LLM_MODEL_CREATE_STANDALONE_QUESTION
    temperature = 0.0

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=200,
        google_api_key=api_key,
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

def llm_classify() -> ChatGoogleGenerativeAI:
    api_key = settings.GEMINI_API_KEY_CREATE_STANDALONE_QUESTION
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY_CREATE_STANDALONE_QUESTION not set")

    model = settings.LLM_MODEL_CREATE_STANDALONE_QUESTION
    temperature = 0.0

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=200,
        google_api_key=api_key,
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