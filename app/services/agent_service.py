from langchain.agents import create_react_agent, AgentExecutor
from app.tools.index import TOOLS
from app.core.llm import llm_plan
from app.config.prompt import get_react_prompt

class AgentService:
    @staticmethod
    def run_agent(question: str):
        llm = llm_plan()
        prompt = get_react_prompt()
        agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
        executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True, handle_parsing_errors=True)

        result = executor.invoke({"input": question})
        # summary = summarize_json(result.get("output"), "travel planning")
        return result
