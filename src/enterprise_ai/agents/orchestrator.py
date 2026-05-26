import logging
import os
import time

import dotenv
from google.oauth2.credentials import Credentials
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent, tool
from strands.agent.agent_result import AgentResult
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

from enterprise_ai.agents.codebase_agent import CodebaseAgent
from enterprise_ai.agents.drive_agent import DriveAgent
from enterprise_ai.agents.tavily_agent import TavilyAgent

dotenv.load_dotenv()

from enterprise_ai.agents.tools import (
    trigger_codebase_agent,
    trigger_google_drive_agent,
    trigger_tavily_agent,
)


class OrchestratorAgent():
    def __init__(self):
        self.logger = logging.getLogger("strands")
        from dotenv import load_dotenv
        load_dotenv()

        # setup LLM
        self.bedrock_model = BedrockModel(model_id=os.getenv("BEDROCK_MODEL_ID"))

        # setup Agent
        self.system_prompt = """
            CRITICAL MANDATORY RULES - FOLLOW EXACTLY

            RULE 1 (ABSOLUTE PRIORITY):
            If the user message contains "drive.google.com" or mentions "Google Drive":
            → YOU MUST call trigger_google_drive_agent(drive_link) FIRST
            → This is NOT optional
            → Extract the full Google Drive URL and pass it to the tool

            RULE 2:
            If the user message contains "github.com":
            → YOU MUST call trigger_codebase_agent(repo_url)

            RULE 3:
            If the user asks for current information or best practices:
            → call trigger_tavily_agent(query)

            RULE 4 (IMPORTANT):
            You can and SHOULD use MULTIPLE tools in a single response.
            Example: If user provides Drive link + asks question → use BOTH drive AND tavily agents

            ═══════════════════════════════════════════

            You are the Orchestrator AI for Nowacoski, an intelligent onboarding agent designed to create personalized, efficient onboarding experiences for full-time employees.

            Your goal is to help the manager successfully onboard a new employee by generating a custom onboarding plan that maximizes context, minimizes time spent by the manager, and ensures the employee becomes productive in their role.

            Use specialized agents to gather context from Google Workspace, online resources, and internal documents. Critically evaluate and prioritize this context to avoid overload and ensure relevance.

            At the end of onboarding, the employee should demonstrate role-specific proficiency. For example, they might be able to: "Build a Slack support bot using Generative AI to assist sales agents."

            Output an actionable plan that includes:
            - A step-by-step onboarding roadmap
            - Key documents and tools to review
            - Suggested coding tasks or deliverables
            - Milestones to assess learning progress
            - Questions for the manager to clarify missing info

            Use memory (MongoDB) to manage context across interactions, and remain aware of system limitations (e.g., token limits).

            Your job is not to just fetch data but to orchestrate the learning journey, adapting dynamically to the employee's evolving understanding and goals.

            Ask for additional context or input from the user where necessary.
        """
        tools = [trigger_tavily_agent, trigger_google_drive_agent, trigger_codebase_agent]
        agent = Agent(
            tools=tools,
            model=self.bedrock_model,
            callback_handler=None
        )
        self.messages = agent.messages
        self.tool_names = sorted(agent.tool_names)

    def __call__(self, message: str) -> AgentResult:
        # an agent is created for each call...
        agent = Agent(
            tools=[trigger_tavily_agent, trigger_google_drive_agent, trigger_codebase_agent],
            messages=self.messages,
            model=self.bedrock_model,
            system_prompt=self.system_prompt,
            callback_handler=self._callback_handler
        )
        response = agent(message)
        self.messages = agent.messages
        return response

    def _callback_handler(self, **kwargs):
        if kwargs.get("reasoning") and "reasoningText" in kwargs:
            self.logger.info(f"Reasoning: {kwargs['reasoningText']}")

if __name__ == "__main__":
    # for testing purposes
    agent = OrchestratorAgent()
    print(agent('ayo'))

# test context boundaries--offload the context to storage
### https://arxiv.org/abs/2310.08560
# optimize knowledge base thru means other than s3
## store session context
