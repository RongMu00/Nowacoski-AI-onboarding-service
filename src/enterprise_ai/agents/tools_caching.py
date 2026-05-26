import logging
import os
import pickle
from typing import Dict, List, Optional

import dotenv
from strands import tool

from enterprise_ai.agents.codebase_agent_caching import CodebaseAgent
from enterprise_ai.agents.drive_agent_caching import DriveAgent
from enterprise_ai.agents.slack_agent_caching import SlackAgent
from enterprise_ai.agents.tavily_agent_caching import TavilyAgent
from enterprise_ai.storage.vector_store import MongoVectorStore, get_vector_store

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Lazy singletons — avoid re-creating expensive objects
# (MongoDB connection + embedding model + MCP client) on every call
# ──────────────────────────────────────────────────────────────

_tavily_agent: Optional[TavilyAgent] = None
_slack_agent: Optional[SlackAgent] = None


def _get_tavily_agent() -> TavilyAgent:
    global _tavily_agent
    if _tavily_agent is None:
        _tavily_agent = TavilyAgent()
    return _tavily_agent


def _get_slack_agent() -> SlackAgent:
    global _slack_agent
    if _slack_agent is None:
        _slack_agent = SlackAgent()
    return _slack_agent


def _get_vector_store() -> MongoVectorStore:
    """Use the shared singleton from vector_store.py — no duplicate connections."""
    return get_vector_store()


# ──────────────────────────────────────────────────────────────
# Raw search helpers (for Plan-Action-Reflect result fusion)
# These return structured data, NOT @tool-decorated.
# ──────────────────────────────────────────────────────────────


def raw_web_search(query: str) -> List[Dict]:
    """Execute web search via Tavily and return structured results for fusion.

    Returns:
        list of {"title": str, "url": str, "content": str}
    """
    logger.info(f"Raw web search: {query[:60]}")
    try:
        agent = _get_tavily_agent()
        return agent.search_raw(query)
    except Exception as e:
        logger.error(f"Raw web search error: {e}")
        return []


def raw_slack_search(channel_id: str, limit: int = 50) -> List[Dict]:
    """Fetch Slack channel messages and return structured results for fusion.

    Returns:
        list of {"title": str, "url": str, "content": str}
    """
    logger.info(f"Raw Slack search: channel {channel_id}")
    try:
        agent = _get_slack_agent()
        return agent.search_raw(channel_id, limit)
    except Exception as e:
        logger.error(f"Raw Slack search error: {e}")
        return []


def raw_vectordb_search(query: str, folder_id: Optional[str] = None) -> List[Dict]:
    """Execute VectorDB semantic search and return structured results for fusion.

    Returns:
        list of {"title": str, "url": str, "content": str}
    """
    logger.info(f"Raw VectorDB search: {query[:60]}")
    try:
        store = _get_vector_store()
        results = store.search_similar(
            query=query, folder_id=folder_id, top_k=5, threshold=0.3
        )
        return [
            {
                "title": r.get("source", "Cached Document"),
                "url": "",
                "content": r.get("content", ""),
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Raw VectorDB search error: {e}")
        return []


@tool
def trigger_tavily_agent(context: str) -> str:
    """
    Triggers the Tavily Agent to perform a web search with VectorDB caching.

    Features:
    - Searches the web for current information
    - Caches results in MongoDB for future queries
    - Returns semantic similarity matches from cache when available

    Args:
        context (str): Natural language search query

    Returns:
        str: Concise summary from web search or cached results

    Example:
        >>> trigger_tavily_agent("Best practices for React in 2024")
    """
    logger.info(f"🌐 Tavily search triggered: {context[:50]}")

    try:
        agent = _get_tavily_agent()
        response = agent(context)
        logger.info("✅ Tavily search completed")
        return str(response)
    except Exception as e:
        logger.error(f"Tavily agent error: {e}")
        return f"❌ Web search error: {str(e)}"


@tool
def trigger_google_drive_agent(drive_links: str) -> str:
    """
    Triggers the Google Drive Agent with VectorDB caching.

    Features:
    - Processes Google Drive folders for onboarding documents
    - Caches documents and embeddings in MongoDB
    - Returns cached results for repeated queries (0.2s instead of 3-5s)
    - Supports Word, Excel, Google Docs, Google Sheets, and text files

    Args:
        drive_links (str): Google Drive folder URL
                          Format: https://drive.google.com/drive/folders/FOLDER_ID

    Returns:
        str: Formatted onboarding plan with caching status

    Example:
        >>> trigger_google_drive_agent(
        ...     'https://drive.google.com/drive/folders/1abc2def3ghi4jkl5mno'
        ... )
    """
    logger.info(f"📁 Google Drive agent triggered with link: {drive_links}")

    try:
        # Initialize credentials
        creds = None
        service_account_path = os.getenv('SERVICE_ACCOUNT_PATH')

        # Try to load user OAuth credentials
        if os.path.exists('token.pickle'):
            try:
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
                logger.info("✅ User OAuth credentials loaded")
            except Exception as e:
                logger.warning(f"Could not load token.pickle: {e}")

        # Initialize DriveAgent with VectorDB
        agent = DriveAgent(
            credentials=creds,
            service_account_path=service_account_path
        )
        logger.info(f"✅ DriveAgent initialized (Auth: {agent.auth_method})")

        # Validate drive link
        if not drive_links or not _is_valid_drive_link(drive_links):
            logger.error(f"Invalid Drive link: {drive_links}")
            return _format_error_response(
                "Invalid Google Drive Link",
                f"Expected format: https://drive.google.com/drive/folders/FOLDER_ID\nReceived: {drive_links}"
            )

        # Create onboarding plan (with caching)
        logger.info(f"Creating onboarding plan from: {drive_links}")
        plan = agent.create_onboarding_plan(drive_links)

        # Format response
        return _format_plan_response(plan, agent.auth_method)

    except Exception as e:
        logger.exception(f"Drive agent error: {e}")
        return _format_error_response("Drive Agent Error", str(e))


@tool
def trigger_codebase_agent(repo_url: str, query: str = None) -> str:
    """
    Triggers the Codebase Agent with VectorDB caching and hybrid deep-dive.

    Features:
    - Analyzes GitHub repositories for developer onboarding
    - Caches analysis results in MongoDB
    - Returns cached results for repeated queries
    - Extracts technologies, setup steps, and learning paths
    - Hybrid mode: when a specific query is provided and the cached analysis
      doesn't cover it, fetches targeted files from GitHub for a deeper answer

    Args:
        repo_url (str): GitHub repository URL
                       Format: https://github.com/username/repository
        query (str, optional): Specific question about the repo.
                              When provided, enables hybrid cache + targeted fetch.

    Returns:
        str: Structured onboarding plan or targeted analysis for the codebase

    Example:
        >>> trigger_codebase_agent(
        ...     'https://github.com/facebook/react'
        ... )
        >>> trigger_codebase_agent(
        ...     'https://github.com/facebook/react',
        ...     query='How does the fiber reconciliation algorithm work?'
        ... )
    """
    logger.info(f"📦 Codebase agent triggered: {repo_url}")
    if query:
        logger.info(f"   Specific query: {query[:80]}")

    try:
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            logger.error("GITHUB_TOKEN not set")
            return "❌ Error: GITHUB_TOKEN environment variable not set"

        agent = CodebaseAgent(github_token)
        logger.info("✅ CodebaseAgent initialized")

        onboarding = agent.create_onboarding_plan(repo_url, query=query)
        logger.info("✅ Codebase analysis completed")

        return str(onboarding)

    except Exception as e:
        logger.error(f"Codebase agent error: {e}")
        return f"❌ Repository analysis error: {str(e)}"


@tool
def trigger_slack_agent(channel_id: str) -> str:
    """
    Triggers the Slack Agent to fetch and analyze channel messages with VectorDB caching.

    Features:
    - Fetches recent messages from a Slack channel
    - Caches messages in MongoDB for future queries
    - Enriches messages with user display names
    - Returns LLM-generated summary of channel activity

    Args:
        channel_id (str): Slack channel ID (e.g., "C0123456789")

    Returns:
        str: Summary of channel messages relevant to onboarding

    Example:
        >>> trigger_slack_agent("C0123456789")
    """
    logger.info(f"💬 Slack agent triggered: {channel_id}")

    try:
        agent = _get_slack_agent()
        response = agent(channel_id)
        logger.info("✅ Slack analysis completed")
        return str(response)
    except Exception as e:
        logger.error(f"Slack agent error: {e}")
        return f"❌ Slack analysis error: {str(e)}"


@tool
def semantic_search_vectordb(query: str, folder_id: Optional[str] = None) -> str:
    """
    Search all cached content semantically using MongoDB VectorDB.

    Features:
    - Searches by meaning, not keywords
    - Can limit search to specific folder
    - Returns results ranked by relevance

    Args:
        query (str): Search query
        folder_id (str, optional): Limit search to specific folder
                                   e.g., 'github-react', 'tavily_searches'

    Returns:
        str: Ranked search results with relevance scores

    Example:
        >>> semantic_search_vectordb("How to set up React development?")
        >>> semantic_search_vectordb("First day activities", folder_id="onboarding_folder")
    """
    logger.info(f"🔍 Semantic search: {query}")

    try:
        vector_store = _get_vector_store()

        results = vector_store.search_similar(
            query=query,
            folder_id=folder_id,
            top_k=5,
            threshold=0.3
        )

        if not results:
            return f"No relevant cached content found for: {query}"

        output = f"📚 **Found {len(results)} relevant cached documents:**\n\n"

        for i, result in enumerate(results, 1):
            similarity = result.get('similarity', 0)
            source = result.get('source', 'Unknown')
            content = result.get('content', '')[:200]

            bar_length = int(similarity * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            output += f"{i}. **{source}**\n"
            output += f"   Relevance: `{bar}` {similarity:.0%}\n"
            output += f"   > {content}...\n\n"

        return output

    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return f"❌ Search error: {e}"


def _is_valid_drive_link(link: str) -> bool:
    """Validate Google Drive folder link format"""
    valid_formats = [
        'drive.google.com/drive/folders/',
        'drive.google.com/open?id=',
    ]
    return any(fmt in link for fmt in valid_formats)


def _format_plan_response(plan, auth_method: str) -> str:
    """Format onboarding plan response with caching info"""
    result = "**📋 Google Drive Folder Analysis**\n\n"

    # Authentication status
    if auth_method == 'service_account':
        result += "✅ *Using Service Account (Always available)*\n\n"
    elif auth_method == 'user_oauth':
        result += "✅ *Using User OAuth Credentials*\n\n"
    else:
        result += "ℹ️ *Using Demo Mode*\n\n"

    # Cache status
    if plan.cached:
        result += "📦 *Using cached documents (from VectorDB)* ⚡\n\n"
    else:
        result += "🔄 *Fresh fetch from Google Drive*\n\n"

    # Accessibility
    if not plan.is_accessible:
        result += "⚠️ **Access Issue:**\n"
        if plan.steps:
            result += f"{plan.steps[0]}\n\n"
        if plan.error_message:
            result += f"**Error:** {plan.error_message}\n\n"
    else:
        if plan.is_mock:
            result += "ℹ️ *Note: Using demo data. Authenticate for real Drive access.*\n\n"

    # Onboarding plan
    result += "**📖 Onboarding Plan:**\n\n"

    if plan.steps:
        for step in plan.steps:
            if step.startswith(('📚', '🛠️', '📖', '👥', '💻', '🎯', '✅', '❌', '⚠️')):
                result += f"{step}\n"
            else:
                result += f"  {step}\n"
    else:
        result += "No steps available\n"

    # Metadata
    result += f"\n**⏱️ Estimated Duration:** {plan.estimated_duration}\n"
    result += f"**📊 Source:** {plan.source}\n"

    if plan.error_message and plan.is_accessible:
        result += f"\n**ℹ️ Note:** {plan.error_message}\n"

    return result


def _format_error_response(title: str, message: str) -> str:
    """Format error response"""
    result = f"**❌ {title}**\n\n"
    result += f"**Details:**\n{message}\n\n"
    result += "**Solutions:**\n"
    result += "1. Check that the link is valid\n"
    result += "2. Make sure the resource is shared with your account\n"
    result += "3. Wait 5 minutes after sharing (permissions take time to sync)\n"
    result += "4. For API access, run: `python authorize_drive.py`\n"
    return result


# Helper function to get cache statistics
def get_vectordb_stats() -> dict:
    """Get MongoDB VectorDB statistics"""
    try:
        store = _get_vector_store()
        return store.get_stats()
    except Exception as e:
        logger.warning(f"Could not get VectorDB stats: {e}")
        return {}


# Helper function to clear cache
def clear_vectordb_cache(folder_id: str) -> int:
    """Clear VectorDB cache for specific folder"""
    try:
        store = _get_vector_store()
        deleted = store.delete_folder_documents(folder_id)
        logger.info(f"Cleared {deleted} documents from {folder_id}")
        return deleted
    except Exception as e:
        logger.error(f"Could not clear cache: {e}")
        return 0
