from langchain.tools import Tool
from app.tools.weather import weather_tool_wrapper
from app.tools.summary import summarize_text

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

summarization_tool = Tool.from_function(
    func=summarize_text,
    name="summarization_tool",
    description=(
        "Summarize or rewrite a travel-related text into a concise, friendly Vietnamese itinerary. "
        "Keep key details like attractions, activities, food, accommodations, and weather. "
        "Present the result in a clear, day-by-day format with a warm, natural tone — "
        "like a local tour guide describing the trip. "
        "Do not add any new or imagined information. "
        "Input: one string containing the original text to summarize."
    )
)
TOOLS = [weather_tool, summarization_tool]