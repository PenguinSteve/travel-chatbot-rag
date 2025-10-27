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
            name="rag_document_retrieval",
            func=lambda tool_input: retrieve_document_rag_wrapper(tool_input,
                                                                  pinecone_repository,
                                                                flashrank_compressor),
            description=(
                "Use this tool to retrieve relevant travel information from the RAG knowledge base. "
                "This tool must be used when the user requests a travel plan, trip, or itinerary.\n\n"
                "You must call this tool sequentially for each topic in the following order:\n"
                "1) Food → 2) Accommodation\n\n"
                "Each call requires a JSON object as input with the following structure:\n\n"
                "{\n"
                '  "topic": "one of [Food, Accommodation]",\n'
                '  "location": "the city or destination mentioned in the user’s request (e.g., Đà Nẵng, Hà Nội, Thành phố Hồ Chí Minh)",\n'
                '  "query": "a short, focused question that combines the topic and location. Examples:\n'
                '      - topic: "Food" → query: "What are the most famous local dishes in Đà Nẵng?"\n'
                '      - topic: "Accommodation" → query: "What are the best hotels to stay in Đà Nẵng?"\n'
                '  }\n\n'
                "You must wait for the observation result of each call before continuing to the next topic."
            ),
        )

        self.TOOLS = TOOLS + [rag_tool]

        agent = create_react_agent(llm=self.llm, tools=self.TOOLS, prompt=self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.TOOLS, verbose=True, handle_parsing_errors=True)


    def run_agent(self, question: str):

        result = self.executor.invoke({"input": question})
        # summary = summarize_json(result.get("output"), "travel planning")
        return result
