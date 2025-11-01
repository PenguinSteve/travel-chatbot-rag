from langchain_core.prompts import ChatPromptTemplate
from app.core.llm import llm_summary

def summarize_text(text: str) -> str:
    llm = llm_summary()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a professional Vietnamese travel editor and local tour guide.

Your task is to summarize and organize all collected trip information 
(food, accommodation, attractions, and weather) into a natural, day-by-day Vietnamese itinerary.

GUIDELINES:
- Write **only** based on the provided information — do **not** invent or assume missing details.
- If the context lacks enough information for a complete itinerary:
  → Respond exactly: "Tôi chưa đủ thông tin để tạo lịch trình chi tiết."
  → Then politely suggest what additional information the user should provide 
    (e.g., địa điểm, thời gian chuyến đi, món ăn, nơi ở...).
- If enough details are available:
  → Create a friendly, realistic itinerary (Ngày 1, Ngày 2, Ngày 3, ...).
  → Each day should include relevant attractions, meals, accommodation, and weather.
  → Use a warm, conversational tone, as if guiding a traveler.
  → Begin naturally (e.g., “Xin chào bạn…” or “Hành trình của bạn sẽ bắt đầu với…”)
    and end with a pleasant closing remark (e.g., “Chúc bạn có một chuyến đi thật đáng nhớ!”).
- Format the output in **Markdown** for readability.
"""
        ),
        (
            "human",
            "Dưới đây là thông tin thu thập được, hãy tóm tắt và tạo lịch trình chuyến đi:\n\n{text}",
        ),
    ])

    chain = prompt | llm
    result = chain.invoke({"text": text})
    return result.content.strip()
