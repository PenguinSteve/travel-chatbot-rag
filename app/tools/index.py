from langchain.tools import Tool
from app.tools.weather import weather_tool_wrapper
from app.tools.summary import summarize_text
from app.tools.rag import retrieve_document_rag_wrapper

weather_tool = Tool.from_function(
    func=weather_tool_wrapper,
    name="weather_forecast",
    description = (
        "Get the daily weather forecast for a specific city and date range. "
        "The input must be a valid JSON object with the following fields:"
        "{"
        '  "city": "name of the city (e.g., Ho Chi Minh, Da Nang, Hanoi)",'
        '  "start_date": "start date in YYYY-MM-DD",'
        '  "end_date": "end date in YYYY-MM-DD"'
        "}"
    )
)

# rag_tool = Tool.from_function(
#     func=retrieve_document_rag_wrapper,
#     name="rag_document_retrieval",
#     description=(
#         "Use this tool to retrieve relevant travel information from the RAG knowledge base. "
#         "This tool must be used when the user requests a travel plan, trip, or itinerary.\n\n"
#         "You must call this tool sequentially for each topic in the following order:\n"
#         "1) Food → 2) Accommodation\n\n"
#         "Each call requires a JSON object as input with the following structure:\n\n"
#         "{\n"
#         '  "topic": "one of [Food, Accommodation]",\n'
#         '  "location": "the city or destination mentioned in the user’s request (e.g., Đà Nẵng, Hà Nội, Thành phố Hồ Chí Minh)",\n'
#         '  "query": "a short, focused question that combines the topic and location. Examples:\n'
#         '      - topic: "Food" → query: "What are the most famous local dishes in Đà Nẵng?"\n'
#         '      - topic: "Accommodation" → query: "What are the best hotels to stay in Đà Nẵng?"\n'
#         '  }\n\n'
#         "You must wait for the observation result of each call before continuing to the next topic."
#     ),
# )

summarization_tool = Tool.from_function(
    func=summarize_text,
    name="summarization_tool",
    description=(
        "Use this tool to summarize or rewrite a travel-related text into a concise, "
        "friendly, and tour-guide-style itinerary in Vietnamese. "
        "The summary must retain key details such as attractions, activities, foods, accommodations, and weather, "
        "while presenting them in a clear, day-by-day format if possible. "
        "The tone should be warm, natural, and engaging — like a local tour guide narrating a trip. "
        "Do not add new or imagined details beyond the input text. "
        "Input: a single string containing the original text to be summarized."
    ),
)

TOOLS = [weather_tool, summarization_tool]