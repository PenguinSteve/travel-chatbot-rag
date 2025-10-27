from app.core.llm import llm_summary

def summarize_text(text: str) -> str:
    llm = llm_summary()
    prompt = f"""
    You are a professional tour guide and travel assistant.
    Please rewrite the following travel-related text in Vietnamese
    into a complete, engaging, and friendly itinerary description — 
    as if you were personally guiding a traveler through the journey.

    Requirements:
    - Keep **all** important details: attractions, activities, local foods, accommodations, and weather.
    - Organize the information **clearly by day or activity** if possible.
    - Use a warm, natural, and conversational tone — friendly, but still informative.
    - Do **not** shorten or omit major details. Focus on clarity and flow instead.
    - Do **not** invent or add new information not present in the text.
    - Write it as if it were a **spoken tour guide narration**, inspiring and easy to follow.

    Example opening style:
    "Xin chào bạn! Hãy cùng tôi khám phá chuyến hành trình thú vị này nhé..."

    ---
    {text}
    """
    response = llm.invoke(prompt)
    return response.content
