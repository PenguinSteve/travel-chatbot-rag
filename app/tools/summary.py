from langchain_core.prompts import ChatPromptTemplate
from app.core.llm import llm_summary

def summarize_text(text: str) -> str:
    llm = llm_summary()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a professional travel editor and Vietnamese tour guide.
    Generate a complete, realistic Vietnamese travel itinerary based on the given content and trip duration.
    Use all relevant details (attractions, food, accommodation, weather).
    Do not invent missing facts. Write warmly and naturally in Vietnamese.
    The output should be a full day-by-day itinerary (Day 1, Day 2, etc.).
    Begin naturally (e.g., "Xin chào bạn...") and end with a friendly closing line.
    """
        ),
        (
            "human",
            "Summarize and create the itinerary based on the following context:\n\n{text}",
        ),
    ])

    chain = prompt | llm
    result = chain.invoke({"text": text})
    print("\n---------------------Generated Summary:---------------------\n")
    print(result.content.strip())
    return result.content.strip()
