from app.core.llm import llm_summary

def summarize_text(text: str) -> str:
    llm = llm_summary()
    prompt = f"""
    You are a professional travel editor and experienced Vietnamese tour guide assistant.

    Your task is to **generate a detailed travel itinerary** based on the provided travel-related content 
    and the user's specified trip duration.

    ### Context:
    {text}

    ### Requirements:
    - Use all relevant information from the input (attractions, activities, local food, accommodations, weather, etc.).
    - The result must be a **complete itinerary** that follows the user's trip duration exactly 
    (e.g., "3 days 2 nights", "5 days 4 nights", etc.).
    - If the duration is not explicitly stated, assume a reasonable length (default: 3 days 2 nights).
    - Clearly divide the itinerary by **Day 1, Day 2, Day 3**, etc.
    - For each day, organize the content as follows:
    • **Morning:** main activities, first attractions to visit  
    • **Noon:** local food or restaurant suggestions  
    • **Afternoon:** sightseeing, relaxing, or cultural experiences  
    • **Evening:** night activities, entertainment, or dining recommendations
    - Maintain a **warm, friendly, and conversational tone**, as if you were a real Vietnamese tour guide speaking to travelers.
    - Do **not** invent or add any information that is not present in the input.
    - Keep the total length around **40–50%** of the original text, but make sure the itinerary feels complete and natural.
    - The final output must be written **entirely in Vietnamese**, with **no English words**, **no Markdown formatting**, 
    and **no metadata**.
    - Begin naturally, for example:
    "Xin chào bạn! Hãy cùng tôi khám phá hành trình thú vị này nhé..."
    - End with a friendly closing sentence, such as:
    "Hy vọng hành trình này sẽ mang đến cho bạn những trải nghiệm thật đáng nhớ nhé!"

    ### Output:
    Return a single, well-structured Vietnamese text that presents a **day-by-day travel itinerary**.
    """


    response = llm.invoke(prompt)
    return response.content
