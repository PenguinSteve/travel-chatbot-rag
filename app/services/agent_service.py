import json
from langchain.agents import create_react_agent, AgentExecutor
from langchain_pinecone import PineconeRerank
from app.models.chat_schema import ChatMessage
from app.repositories.chat_repository import ChatRepository
from langchain.retrievers import ParentDocumentRetriever
from app.tools.index import TOOLS
from app.core.llm import llm_plan
from app.config.prompt import get_react_prompt

from bson.objectid import ObjectId
from langchain.tools import Tool
from app.tools.rag import retrieve_document_rag_wrapper
from app.tools.schedule import schedule_trip


class AgentService:
    def __init__(self, retriever: ParentDocumentRetriever, chat_repository: ChatRepository, pinecone_reranker: PineconeRerank, user_id: str):
        self.llm = llm_plan()
        self.chat_repository = chat_repository
        self.prompt = get_react_prompt()
        self.retriever = retriever
        self.pinecone_reranker = pinecone_reranker
        self.user_id = user_id

        rag_tool = Tool(
            name="retrieve_document_rag",
            func=lambda tool_input: retrieve_document_rag_wrapper(tool_input, retriever=self.retriever, pinecone_reranker=self.pinecone_reranker),
            description=(
                "Retrieve detailed travel information from the RAG knowledge base for a given topic "
                "in one of the supported cities (Thành phố Hồ Chí Minh, Đà Nẵng, or Hà Nội). "
                "Use this tool to collect accurate local data about food, accommodations "
                "before generating the trip itinerary. "
                "Input must be a JSON object in the following format: "
                "{ "
                '"topic": ["Food"] | ["Accommodation"] | ["Food", "Accommodation"], '
                '"location": ["Supported city or mapped district name"], '
                '"query": "Short, focused question combining topic and location" '
                "}. "
                "Output returns relevant travel content and metadata for that topic."
            )
        )

        wrapped_schedule_tool = Tool.from_function(
            func=lambda trip_details_str: self.schedule_trip_wrapper(trip_details_str),
            name="schedule_tool",
            description=(
                "Create a new travel schedule and save it to MongoDB. "
                "Input must be a JSON object containing details such as location, "
                "start_date, end_date, itinerary, weather_summary, and accommodation, tips."
                "The user_id will be added automatically by the system."
            )
        )
        
        original_tools = TOOLS

        final_tools = [tool for tool in original_tools if tool.name != "schedule_tool"]

        final_tools.extend([rag_tool, wrapped_schedule_tool])

        self.TOOLS = final_tools

        agent = create_react_agent(llm=self.llm, tools=self.TOOLS, prompt=self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.TOOLS, verbose=True, handle_parsing_errors=True)


    def run_agent(self, session_id: str, question: str):
        result = self.executor.invoke({"input": question})

        raw_output = result.get('output')

        decoded_answer = None         
        ai_message_for_history = "" 

        try:
            decoder = json.JSONDecoder()
            decoded_answer, _ = decoder.raw_decode(raw_output)
            
            if isinstance(decoded_answer, dict):
                ai_message_for_history = decoded_answer.get('message', raw_output)
            else:
                ai_message_for_history = raw_output
                decoded_answer = raw_output

        except json.JSONDecodeError:
            ai_message_for_history = raw_output  # Dùng chuỗi thô (bị cắt) để lưu
            decoded_answer = raw_output


        self.chat_repository.save_message(session_id=session_id, message=ChatMessage(content=question, role="human"))
        self.chat_repository.save_message(session_id=session_id, message=ChatMessage(content=ai_message_for_history, role="ai"))
    
        return decoded_answer
    
    def schedule_trip_wrapper(self, trip_details_str: str):
        print(f"\n--- Wrapping schedule_tool for user: {self.user_id} ---\n")
        try:
            # 1. Parse JSON mà LLM cung cấp
            trip_details = json.loads(trip_details_str)
            
            # 2. Tiêm user_id (đã lưu trong self)
            trip_details["user_id"] = self.user_id

            generated_trip_id = str(ObjectId())
            trip_details["trip_id"] = generated_trip_id

            schedule_trip(trip_details) 
            
            return trip_details 
        
        except json.JSONDecodeError as e:
            return f"Lỗi: JSON không hợp lệ. {e}"
        except Exception as e:
            return f"Lỗi khi lưu lịch trình: {e}"
