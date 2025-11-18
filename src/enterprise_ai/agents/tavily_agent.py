import logging
import os

from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient


class TavilyAgent():
    def __init__(self):
        self.logger = logging.getLogger("strands")
        from dotenv import load_dotenv
        load_dotenv()

        url='https://mcp.tavily.com/mcp/?tavilyApiKey='+os.getenv('TAVILY_KEY')
        self.mcp_client = MCPClient(lambda: streamablehttp_client(url=url))
        self.bedrock_model = BedrockModel(model_id=os.getenv("BEDROCK_MODEL_ID"))

        # initial agent setup
        self.system_prompt = """
            You are the Tavily Agent, a focused external research assistant in the Nowacoski onboarding system.

            Your role is to search the web based on a clear query provided by the orchestrator and return a concise, reliable summary that directly answers the query or fills the knowledge gap.

            Prioritize:
            - Up-to-date, authoritative sources (official docs, recent tutorials, trusted blogs)
            - Concise summaries with only the most relevant points
            - Linking to original sources when helpful

            Do not hallucinate answers or interpret vague intent — only act on the query as given.

            Return structured, useful output to help the orchestrator onboard the employee more effectively.
        """
        with self.mcp_client:
            tools = self.mcp_client.list_tools_sync()
            agent = Agent(
                tools=tools,
                model=self.bedrock_model,
                system_prompt=self.system_prompt
            )
            self.messages = agent.messages
            self.tool_names = sorted(agent.tool_names)

    def __call__(self, message: str) -> AgentResult:
        # an agent is created for each call...
        with self.mcp_client:
            tools = self.mcp_client.list_tools_sync()
            agent = Agent(
                tools=tools,
                messages=self.messages,
                model=self.bedrock_model,
                system_prompt=self.system_prompt
            )
            print('tavily agent triggered')
            response = agent(message)
            self.messages = agent.messages
            return response
