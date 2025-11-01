from langchain_core.prompts import ChatPromptTemplate
from app.core.llm import llm_summary

def summarize_text(text: str) -> str:
    llm = llm_summary()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a **professional travel editor**.

    Your task is to **summarize the provided content** into a **travel plan or itinerary summary** in English.  
    Include key details such as **location, food, accommodation, weather, and activities** if available.

    Requirements:
    - Use only the given information — do NOT invent details.
    - Write clearly and naturally in a friendly tone.
    - If information is limited, produce a short, meaningful summary.
    - If possible, organize the plan by days (Day 1, Day 2, etc.).

    Output: A short, readable travel summary or itinerary in **English**.
    """
        ),
        (
            "human",
            "Here is the information to summarize:\n\n{text}",
        ),
    ])



    chain = prompt | llm
    result = chain.invoke({"text": text})
    return result.content.strip()
