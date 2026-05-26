import logging
import os
from typing import Dict, List

from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

logger = logging.getLogger("tavily_agent")


class TavilyAgent:
    """Pure worker agent for web search via Tavily.

    This agent only fetches and returns results — it does NOT manage its own
    cache.  All cache-vs-fetch decisions are made by the orchestrator.
    """

    def __init__(self):
        self.logger = logging.getLogger("tavily_agent")

        from dotenv import load_dotenv
        load_dotenv()

        self.tavily_key = os.getenv('TAVILY_KEY', '')

        # Initialize MCP for Tavily
        url = 'https://mcp.tavily.com/mcp/?tavilyApiKey=' + os.getenv('TAVILY_KEY', '')
        self.mcp_client = MCPClient(lambda: streamablehttp_client(url=url))
        self.bedrock_model = BedrockModel(model_id=os.getenv("BEDROCK_MODEL_ID"))

        self.system_prompt = """
            You are the Tavily Agent, a focused external research assistant in the Nowacoski onboarding system.

            Your role is to search the web based on a clear query provided by the orchestrator and return a concise, reliable summary that directly answers the query or fills the knowledge gap.

            Prioritize:
            - Up-to-date, authoritative sources (official docs, recent tutorials, trusted blogs)
            - Concise summaries with only the most relevant points
            - Linking to original sources when helpful

            Do not hallucinate answers or interpret vague intent – only act on the query as given.

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

    def __call__(self, message: str):
        """Perform web search via Tavily MCP and return LLM-summarized result.

        Always fetches fresh — the orchestrator decides whether to call this
        or use cached content.

        Returns:
            AgentResult from Tavily search
        """
        self.logger.info(f"Tavily search: {message}")

        with self.mcp_client:
            tools = self.mcp_client.list_tools_sync()
            agent = Agent(
                tools=tools,
                messages=self.messages,
                model=self.bedrock_model,
                system_prompt=self.system_prompt
            )

            response = agent(message)
            self.messages = agent.messages
            return response

    def search_raw(self, query: str) -> List[Dict]:
        """Return raw web search results as structured data for fusion.

        Uses tavily-python for direct API call (bypasses MCP + LLM
        summarization), returning structured snippets suitable for
        result fusion in the Plan-Action-Reflect loop.

        Returns:
            list of {"title": str, "url": str, "content": str}
        """
        self.logger.info(f"Tavily raw search: {query}")

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.tavily_key)
            response = client.search(query=query, max_results=5)

            snippets = []
            for result in response.get("results", []):
                snippets.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                })

            self.logger.info(f"Tavily raw search returned {len(snippets)} results")
            return snippets

        except Exception as e:
            self.logger.error(f"Tavily raw search error: {e}")
            return []
