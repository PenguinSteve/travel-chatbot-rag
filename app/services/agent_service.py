from langchain.agents import create_react_agent, AgentExecutor
from app.models.chat_schema import ChatMessage
from app.repositories.chat_repository import ChatRepository
from app.repositories.pinecone_repository import PineconeRepository
from langchain_community.document_compressors import FlashrankRerank
from app.tools.index import TOOLS
from app.core.llm import llm_plan
from app.config.prompt import get_react_prompt

from langchain.tools import Tool
import time
from app.tools.rag import retrieve_document_rag_wrapper


class AgentService:
    def __init__(self, chat_repository: ChatRepository, pinecone_repository: PineconeRepository, flashrank_compressor: FlashrankRerank):
        self.llm = llm_plan()
        self.chat_repository = chat_repository
        self.prompt = get_react_prompt()

        rag_tool = Tool(
            name="retrieve_document_rag",
            func=lambda tool_input: retrieve_document_rag_wrapper(tool_input,
                                                                  pinecone_repository,
                                                                flashrank_compressor),
            description=(
                "Retrieve detailed travel information from the RAG knowledge base for a given topic "
                "in one of the supported cities (Thành phố Hồ Chí Minh, Đà Nẵng, or Hà Nội). "
                "Use this tool to collect accurate local data about food, accommodations "
                "before generating the trip itinerary. "
                "Input must be a JSON object in the following format: "
                "{ "
                '"topic": "Food" | "Accommodation"'
                '"location": "Supported city or mapped district name", '
                '"query": "Short, focused question combining topic and location" '
                "}. "
                "Output returns relevant travel content and metadata for that topic."
            )

        )
        

        self.TOOLS = TOOLS + [rag_tool]

        agent = create_react_agent(llm=self.llm, tools=self.TOOLS, prompt=self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.TOOLS, verbose=True, handle_parsing_errors=True)


    def run_agent(self, session_id: str, question: str):
        result = self.executor.invoke({"input": question})
        self.chat_repository.save_message(session_id=session_id, message=ChatMessage(content=question, role="human"))
        self.chat_repository.save_message(session_id=session_id, message=ChatMessage(content=result['output'], role="ai"))
        return result
