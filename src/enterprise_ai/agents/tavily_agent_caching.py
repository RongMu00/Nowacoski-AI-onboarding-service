import logging
import os
from typing import Dict, List, Optional

from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

# Import VectorDB for caching
from enterprise_ai.storage.vector_store import MongoVectorStore

logger = logging.getLogger("tavily_agent")


class TavilyAgent:
    def __init__(self):
        """Initialize TavilyAgent with VectorDB caching"""
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

        # NEW: Initialize VectorDB for caching search results
        try:
            self.vector_store = MongoVectorStore()
            self.logger.info("Vector store initialized for search caching")
        except Exception as e:
            self.logger.warning(f"Vector store not available: {e}")
            self.vector_store = None

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
        """
        Process search query with VectorDB caching

        Flow:
        1. Check cache for similar queries
        2. If no match, search via Tavily
        3. Cache results for future queries

        Returns:
            AgentResult from fresh search, or str from cache
        """
        self.logger.info(f"Tavily search: {message}")

        # STEP 1: Check cache for similar queries
        cached_result = self._get_cached_search(message)
        if cached_result:
            self.logger.info("📦 Using cached search result")
            return cached_result

        # STEP 2: Perform web search via Tavily
        self.logger.info("📡 Fetching from Tavily web search...")

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

            # STEP 3: Cache the result
            if self.vector_store:
                self._cache_search_result(message, response)

            return response

    def search_raw(self, query: str) -> List[Dict]:
        """Return raw web search results as structured data for fusion.

        Unlike __call__() which runs results through an LLM for summarization,
        this method returns structured snippets directly from Tavily API,
        suitable for result fusion in the Plan-Action-Reflect loop.

        Returns:
            list of {"title": str, "url": str, "content": str}
        """
        self.logger.info(f"Tavily raw search: {query}")

        # Check cache first
        cached = self._get_cached_search(query)
        if cached:
            self.logger.info("Using cached search result for raw search")
            return [{"title": "Cached Result", "url": "", "content": cached}]

        # Use tavily-python for direct API call (bypasses MCP + LLM summarization)
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

            # Cache the combined result for future queries
            if self.vector_store and snippets:
                combined_text = "\n\n".join(
                    [f"[{s['title']}] {s['content']}" for s in snippets]
                )
                self._cache_raw_search_result(query, combined_text)

            self.logger.info(f"Tavily raw search returned {len(snippets)} results")
            return snippets

        except Exception as e:
            self.logger.error(f"Tavily raw search error: {e}")
            return []

    def _cache_raw_search_result(self, query: str, combined_text: str) -> bool:
        """Cache raw search result text for future queries."""
        if not self.vector_store:
            return False

        try:
            doc_id = self.vector_store.store_document(
                content=f"Query: {query}\n\nResult:\n{combined_text}",
                source=f"Web Search: {query[:50]}",
                folder_id="tavily_searches",
                metadata={
                    "type": "web_search",
                    "query": query,
                    "query_length": len(query),
                    "result_length": len(combined_text),
                },
            )
            if doc_id:
                self.logger.info(f"Cached raw search result: {doc_id}")
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Could not cache raw search result: {e}")
            return False

    def _get_cached_search(self, query: str) -> Optional[str]:
        """Retrieve cached search results using semantic similarity"""
        if not self.vector_store or self.vector_store.collection is None:
            return None

        try:
            # Search for similar queries
            results = self.vector_store.search_similar(
                query=query,
                folder_id="tavily_searches",
                top_k=1,
                threshold=0.7  # High threshold for similar queries
            )

            if results:
                self.logger.info(f"Found similar cached search (similarity: {results[0]['similarity']:.0%})")
                return results[0]['content']

            return None

        except Exception as e:
            self.logger.warning(f"Could not retrieve cached search: {e}")
            return None

    def _cache_search_result(self, query: str, result: AgentResult) -> bool:
        """Cache search result for future queries"""
        if not self.vector_store:
            return False

        try:
            result_text = str(result)

            doc_id = self.vector_store.store_document(
                content=f"Query: {query}\n\nResult:\n{result_text}",
                source=f"Web Search: {query[:50]}",
                folder_id="tavily_searches",
                metadata={
                    "type": "web_search",
                    "query": query,
                    "query_length": len(query),
                    "result_length": len(result_text)
                }
            )

            if doc_id:
                self.logger.info(f"Cached search result: {doc_id}")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Could not cache search result: {e}")
            return False

    def search_cache(self, query: str, top_k: int = 5) -> str:
        """Search cached web search results semantically"""
        if not self.vector_store:
            return "Vector store not available"

        try:
            results = self.vector_store.search_similar(
                query=query,
                folder_id="tavily_searches",
                top_k=top_k,
                threshold=0.3
            )

            if not results:
                return f"No cached searches found for: {query}"

            output = f"**Found {len(results)} related web searches:**\n\n"

            for i, result in enumerate(results, 1):
                similarity = result.get('similarity', 0)
                source = result.get('source', 'Unknown')
                metadata = result.get('metadata', {})
                original_query = metadata.get('query', 'Unknown')

                output += f"{i}. **{source}** ({similarity:.0%} relevant)\n"
                output += f"   Original query: {original_query}\n"
                output += f"   > {result['content'][:200]}...\n\n"

            return output

        except Exception as e:
            self.logger.error(f"Search cache error: {e}")
            return f"Search error: {e}"

    def get_cache_stats(self) -> Dict:
        """Get statistics about cached searches"""
        if not self.vector_store:
            return {}

        try:
            # Get all documents from tavily_searches folder
            docs = self.vector_store.get_documents_by_folder("tavily_searches", limit=1000)

            return {
                "total_cached_searches": len(docs),
                "average_query_length": sum(
                    d.get('metadata', {}).get('query_length', 0) for d in docs
                ) / max(len(docs), 1)
            }
        except Exception as e:
            self.logger.warning(f"Could not get cache stats: {e}")
            return {}

    def clear_search_cache(self) -> int:
        """Clear all cached search results"""
        if self.vector_store:
            deleted = self.vector_store.delete_folder_documents("tavily_searches")
            self.logger.info(f"Cleared {deleted} cached search results")
            return deleted
        return 0
