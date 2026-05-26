import logging
import os
from typing import Dict, List, Optional

import dotenv
from strands import Agent, tool
from strands.agent.agent_result import AgentResult
from strands.models import BedrockModel

from enterprise_ai.agents.codebase_agent import CodebaseAgent
from enterprise_ai.agents.drive_agent import DriveAgent
from enterprise_ai.agents.tavily_agent import TavilyAgent
from enterprise_ai.agents.tools import (
    trigger_codebase_agent,
    trigger_google_drive_agent,
    trigger_tavily_agent,
)
from enterprise_ai.storage.vector_store import MongoVectorStore

dotenv.load_dotenv()

logger = logging.getLogger("orchestrator")


def _create_vector_tools(vector_store: Optional[MongoVectorStore]):
    """Create @tool-decorated functions that close over the vector_store instance.

    This avoids using @tool on instance methods, which can expose 'self'
    as a tool parameter to the LLM.
    """

    @tool
    def semantic_search_all(query: str) -> str:
        """
        Search all cached documents semantically

        Searches across:
        - Google Drive documents (onboarding content)
        - GitHub repositories (codebase analyses)
        - Tavily search results (web searches)

        Use this to find relevant cached content without re-fetching.
        """
        logger.info(f"Semantic search: {query}")

        if not vector_store:
            return "Vector store not available"

        try:
            results = vector_store.search_similar(query=query, top_k=5, threshold=0.3)

            if not results:
                return f"No relevant cached content found for: {query}"

            output = f"**Found {len(results)} relevant cached documents:**\n\n"

            for i, result in enumerate(results, 1):
                similarity = result.get('similarity', 0)
                source = result.get('source', 'Unknown')
                metadata = result.get('metadata', {})
                content_snippet = result.get('content', '')[:150]

                # Determine type
                doc_type = "Document"
                if metadata.get('type') == 'codebase_analysis':
                    doc_type = "Repository"
                elif metadata.get('type') == 'web_search':
                    doc_type = "Web Search"

                output += f"{i}. {doc_type}: **{source}** ({similarity:.0%} relevant)\n"
                output += f"   > {content_snippet}...\n\n"

            return output

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return f"Search error: {e}"

    @tool
    def get_cache_status() -> str:
        """
        Get statistics about cached content across all sources

        Shows:
        - Total cached documents
        - Cache breakdown by source
        - Embedding model info
        """
        logger.info("Retrieving cache status")

        if not vector_store:
            return "Vector store not available"

        try:
            stats = vector_store.get_stats()

            output = "**Vector Cache Status:**\n\n"
            output += f"- Total Cached Documents: {stats.get('document_count', 0)}\n"
            output += f"- Embedding Dimension: {stats.get('embedding_dimension', 0)}\n"
            output += f"- Database: {stats.get('database', 'Unknown')}\n"
            output += f"- Collection: {stats.get('collection', 'Unknown')}\n"
            output += "\n**Cache Sources:**\n"
            output += "- Google Drive (onboarding_content)\n"
            output += "- GitHub (github-*)\n"
            output += "- Web Searches (tavily_searches)\n"

            return output

        except Exception as e:
            logger.warning(f"Could not get cache status: {e}")
            return "Unable to retrieve cache statistics"

    return semantic_search_all, get_cache_status


class OrchestratorAgent:
    def __init__(self):
        """Initialize Orchestrator with multi-agent coordination and VectorDB"""
        self.logger = logging.getLogger("orchestrator")

        # Initialize Bedrock model
        self.bedrock_model = BedrockModel(model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"))

        # NEW: Initialize VectorDB for orchestration memory
        try:
            self.vector_store = MongoVectorStore()
            self.logger.info("Vector store initialized for orchestrator")
        except Exception as e:
            self.logger.warning(f"Vector store not available: {e}")
            self.vector_store = None

        # Initialize sub-agents
        self.drive_agent = None
        self.tavily_agent = None
        self.codebase_agent = None

        # Setup system prompt
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

            ═══════════════════════════════════════════════════════════

            You are the Orchestrator AI for Nowacoski, an intelligent onboarding agent designed to create personalized, efficient onboarding experiences for full-time employees.

            Your capabilities:
            1. **Google Drive Integration** - Fetch and cache onboarding documents (with VectorDB)
            2. **GitHub Analysis** - Analyze codebases and repositories (with VectorDB caching)
            3. **Web Search** - Get current information via Tavily (with search result caching)
            4. **Semantic Search** - Search all cached content by meaning
            5. **Intelligent Routing** - Use the right agent for each query

            Your goal is to help the manager successfully onboard a new employee by:
            - Gathering context from Google Workspace, online resources, and code repositories
            - Using cached data when available (faster and cheaper!)
            - Critically evaluating and prioritizing context
            - Generating a custom onboarding plan

            Output an actionable plan that includes:
            - Step-by-step onboarding roadmap
            - Key documents and tools to review
            - Suggested coding tasks or deliverables
            - Milestones to assess learning progress
            - Questions for the manager to clarify missing info

            ALWAYS leverage the vector database for:
            - Caching documents and analyses
            - Semantic search across all content
            - Reducing redundant API calls
            - Improving response speed
        """

        # Create vector-aware tools via factory (avoids @tool on instance methods)
        self._semantic_search_all, self._get_cache_status = _create_vector_tools(self.vector_store)

        # Define tools
        tools = [
            trigger_tavily_agent,
            trigger_google_drive_agent,
            trigger_codebase_agent,
            self._semantic_search_all,
            self._get_cache_status
        ]

        agent = Agent(
            tools=tools,
            model=self.bedrock_model,
            callback_handler=None
        )
        self.messages = agent.messages
        self.tool_names = sorted(agent.tool_names)

    def __call__(self, message: str) -> AgentResult:
        """Process user message with intelligent agent routing"""
        self.logger.info(f"Orchestrator received: {message[:100]}")

        agent = Agent(
            tools=[
                trigger_tavily_agent,
                trigger_google_drive_agent,
                trigger_codebase_agent,
                self._semantic_search_all,
                self._get_cache_status
            ],
            messages=self.messages,
            model=self.bedrock_model,
            system_prompt=self.system_prompt,
            callback_handler=self._callback_handler
        )

        response = agent(message)
        self.messages = agent.messages

        return response

    def _callback_handler(self, **kwargs):
        """Handle agent callbacks"""
        if kwargs.get("reasoning") and "reasoningText" in kwargs:
            self.logger.info(f"Reasoning: {kwargs['reasoningText']}")

    def get_cache_stats(self) -> Dict:
        """Get detailed cache statistics"""
        if self.vector_store:
            return self.vector_store.get_stats()
        return {}

    def clear_cache(self, folder_id: Optional[str] = None) -> str:
        """
        Clear cache for specific folder or all caches

        Args:
            folder_id: Optional folder ID. If not provided, prompt user.
        """
        if not self.vector_store:
            return "Vector store not available"

        if folder_id:
            deleted = self.vector_store.delete_folder_documents(folder_id)
            return f"Cleared {deleted} documents from folder {folder_id}"
        else:
            return "Please specify a folder_id to clear. Options: 'github-repo-name', 'tavily_searches', or your Drive folder ID"


if __name__ == "__main__":
    # For testing purposes
    agent = OrchestratorAgent()

    # Example: Test with Drive link
    # response = agent('Process this onboarding folder: https://drive.google.com/drive/folders/1abc2def3ghi')

    # Example: Test with GitHub
    # response = agent('Analyze this codebase: https://github.com/username/repository')

    # Example: Test with web search
    # response = agent('What are best practices for React in 2024?')

    print("Orchestrator initialized with VectorDB caching enabled")
