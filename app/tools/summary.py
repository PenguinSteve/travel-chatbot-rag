from langchain_core.prompts import ChatPromptTemplate
from app.core.llm import llm_summary

def summarize_text(text: str) -> str:
    llm = llm_summary()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a professional travel editor and Vietnamese tour guide.

    Your task is to generate a realistic and natural Vietnamese travel itinerary 
    based strictly on the provided context and trip duration.

    INSTRUCTIONS:
    - Only use details that appear in the provided content.
    - Do NOT invent, guess, or assume missing facts.
    - If the information is insufficient to form a complete itinerary:
        → Respond clearly: "Tôi chưa đủ thông tin để tạo lịch trình chi tiết."
        → Then politely suggest what additional information the user should provide 
        (for example: thời gian chuyến đi, địa điểm cụ thể, món ăn, nơi lưu trú, hoạt động mong muốn...).
    - If sufficient information is available:
        → Write a warm and natural Vietnamese travel itinerary (Day 1, Day 2, etc.)
        → Include attractions, food, accommodation, and weather when relevant.
        → Begin naturally (e.g., "Xin chào bạn...") and end with a friendly closing line.
    """
        ),
        (
            "human",
            "Dựa trên thông tin sau, hãy tóm tắt và tạo lịch trình chuyến đi:\n\n{text}",
        ),
    ])

    chain = prompt | llm
    result = chain.invoke({"text": text})
    return result.content.strip()
