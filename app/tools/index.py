from langchain.tools import Tool
from app.tools.weather import weather_tool_wrapper
from app.tools.summary import summarize_text

weather_tool = Tool.from_function(
    func=weather_tool_wrapper,
    name="weather_forecast",
    description = (
        "Retrieve a 3–7 day weather forecast for a specific city and date range. "
        "Use this tool to get weather details relevant to a planned trip itinerary. "
        "Input must be a JSON object with the following fields: "
        "{ "
        '"city": "Name of the supported city (Ho Chi Minh, Da Nang, or Hanoi)", '
        '"start_date": "Start date in YYYY-MM-DD (optional — defaults to today)", '
        '"end_date": "End date in YYYY-MM-DD (optional — defaults to 3 days after start_date)" '
        "}. "
        "If no date range is provided, the tool automatically retrieves a 3-day forecast starting from today. "
        "Output returns temperature, conditions, and other daily forecast details."
    )
)

summarization_tool = Tool.from_function(
    func=summarize_text,
    name="summarization_tool",
    description=(
            "Combine all gathered trip data (food, accommodation, attractions, and weather) "
            "into one friendly, concise Vietnamese travel itinerary. "
            "Input: one long text string containing the collected travel information. "
            "Output: a natural, day-by-day trip summary in Markdown — "
            "realistic, informative, and written in the tone of a local tour guide. "
            "Do not invent new information or add facts not present in the input."
        )
)
TOOLS = [weather_tool, summarization_tool]