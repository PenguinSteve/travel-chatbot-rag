from langchain.agents import create_react_agent, AgentExecutor
from app.repositories.pinecone_repository import PineconeRepository
from langchain_community.document_compressors import FlashrankRerank
from app.tools.index import TOOLS
from app.core.llm import llm_plan
from app.config.prompt import get_react_prompt

from langchain.tools import Tool
from app.tools.rag import retrieve_document_rag_wrapper


class AgentService:
    def __init__(self, pinecone_repository: PineconeRepository, flashrank_compressor: FlashrankRerank):
        self.llm = llm_plan()
        self.prompt = get_react_prompt()

        rag_tool = Tool(
            name="retrieve_document_rag",
            func=lambda tool_input: retrieve_document_rag_wrapper(tool_input,
                                                                  pinecone_repository,
                                                                flashrank_compressor),
           description=(
                "Retrieve relevant travel information from the RAG knowledge base. "
                "This tool is used when the user requests a travel plan, trip, or itinerary. "
                "Each call must include a valid JSON object as input with the following structure:"
                "{"
                '  "topic": "one of [Food, Accommodation]",'
                '  "location": "the city or destination mentioned in the user’s question — must be one of [Đà Nẵng, Hà Nội, Thành phố Hồ Chí Minh]",'
                '  "query": "a short, focused question combining the topic and location. Examples:'
                '      - topic: \\"Food\\" → query: \\"What are the most famous local dishes in Đà Nẵng?\\"'
                '      - topic: \\"Accommodation\\" → query: \\"What are the best hotels to stay in Đà Nẵng?\\"'
                "}"
                "You must always wait for the observation result of each call before moving to the next topic."
            )
        )

        self.TOOLS = TOOLS + [rag_tool]

        agent = create_react_agent(llm=self.llm, tools=self.TOOLS, prompt=self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.TOOLS, verbose=True, handle_parsing_errors=True)


    def run_agent(self, question: str):

        result = self.executor.invoke({"input": question})
        return result
