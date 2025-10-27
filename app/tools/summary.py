from app.core.llm import llm_summary

def summarize_text(text: str) -> str:
    llm = llm_summary()
    prompt = f"""
    You are a professional travel editor and tour guide assistant.

    Your task is to **summarize** the following travel-related text in **Vietnamese**, 
    making it concise but still informative and natural.

    ### Requirements:
    - Keep only the most important information about:
    • attractions, activities, food, accommodations, and weather.
    - Remove repetitive or overly detailed descriptions.
    - Organize the information clearly by **day** or **topic** if possible.
    - Maintain a **warm, friendly, and conversational** tone, like a real Vietnamese tour guide.
    - Do **not** invent or add any new details not present in the input.
    - Aim for about **30–40%** of the original text length.
    - The final output **must be entirely in Vietnamese**.
    - Do **not** include any English explanations, metadata, or markdown formatting.
    - Start naturally, for example:
    "Xin chào bạn! Hãy cùng tôi điểm qua hành trình thú vị này nhé..."

    ### Input text:
    {text}
    """

    response = llm.invoke(prompt)
    return response.content
