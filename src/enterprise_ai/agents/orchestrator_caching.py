import json
import logging
import os
import re
from typing import Dict, List, Optional

import dotenv
from strands import Agent, tool
from strands.agent.agent_result import AgentResult
from strands.models import BedrockModel

from enterprise_ai.agents.tools_caching import (
    raw_slack_search,
    raw_vectordb_search,
    raw_web_search,
    trigger_codebase_agent,
    trigger_google_drive_agent,
    trigger_slack_agent,
    trigger_tavily_agent,
)
from enterprise_ai.storage.vector_store import get_vector_store, MongoVectorStore

dotenv.load_dotenv()

logger = logging.getLogger("orchestrator")


# ──────────────────────────────────────────────────────────────
# Helper: extract JSON from LLM response (strips markdown fences)
# ──────────────────────────────────────────────────────────────

def _extract_json(text: str):
    """Extract JSON array or null from LLM response text.

    Handles markdown code fences, leading/trailing whitespace,
    and the literal string 'null' / 'None'.
    """
    cleaned = text.strip()

    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    if cleaned.lower() in ("null", "none", ""):
        return None

    # Find the first '[' to last ']' span
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing the whole string
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────────────────────────
# ConversationMemory: tracks sources across turns
# ──────────────────────────────────────────────────────────────

class ConversationMemory:
    """Tracks data sources and context across conversation turns.

    Persists as long as the OrchestratorAgent instance lives
    (which is per Streamlit session via st.session_state.agent).

    Stores:
    - repo_urls: GitHub repository URLs discussed
    - drive_folders: Google Drive folder URLs discussed
    - slack_channels: Slack channel IDs discussed
    - turn_summaries: brief summary of what each turn was about
    """

    def __init__(self):
        self.repo_urls: List[str] = []
        self.drive_folders: List[str] = []
        self.slack_channels: List[str] = []
        self.turn_summaries: List[str] = []

    def extract_sources_from_message(self, message: str) -> None:
        """Parse user message for URLs and IDs, store new ones."""
        # GitHub URLs
        for match in re.finditer(
            r"https?://github\.com/[\w\-]+/[\w\-]+", message
        ):
            url = match.group(0).rstrip("/")
            if url not in self.repo_urls:
                self.repo_urls.append(url)
                logger.info(f"Memory: added repo {url}")

        # Google Drive folder URLs
        for match in re.finditer(
            r"https?://drive\.google\.com/drive/folders/[\w\-]+", message
        ):
            url = match.group(0)
            if url not in self.drive_folders:
                self.drive_folders.append(url)
                logger.info(f"Memory: added drive folder {url}")

        # Slack channel IDs (C followed by alphanumeric)
        for match in re.finditer(r"\b(C[A-Z0-9]{8,})\b", message):
            channel_id = match.group(1)
            if channel_id not in self.slack_channels:
                self.slack_channels.append(channel_id)
                logger.info(f"Memory: added slack channel {channel_id}")

    def record_sources_from_results(self, memory: List[Dict]) -> None:
        """After Action phase, extract sources discovered via semantic_search."""
        for m in memory:
            for r in m.get("results", []):
                meta = r.get("metadata", {})
                raw_repo = meta.get("repo_url", "")
                if raw_repo:
                    # Validate: extract a clean github.com URL even if
                    # the metadata contains extra text around it
                    match = re.search(
                        r"https?://github\.com/[\w\-]+/[\w\-]+", raw_repo
                    )
                    if match:
                        clean_url = match.group(0).rstrip("/")
                        if clean_url not in self.repo_urls:
                            self.repo_urls.append(clean_url)
                            logger.info(
                                f"Memory: discovered repo {clean_url} from cache"
                            )
                channel_id = meta.get("channel_id", "")
                if channel_id and channel_id not in self.slack_channels:
                    self.slack_channels.append(channel_id)
                    logger.info(
                        f"Memory: discovered channel {channel_id} from cache"
                    )

    def add_turn_summary(self, summary: str) -> None:
        """Record what a turn was about."""
        self.turn_summaries.append(summary)

    def get_context_for_planner(self) -> str:
        """Format memory as context string for the Plan/Reflect prompts."""
        lines = []

        if self.repo_urls:
            lines.append(
                "Previously discussed GitHub repositories:\n"
                + "\n".join(f"  - {url}" for url in self.repo_urls)
            )
        if self.drive_folders:
            lines.append(
                "Previously discussed Google Drive folders:\n"
                + "\n".join(f"  - {url}" for url in self.drive_folders)
            )
        if self.slack_channels:
            lines.append(
                "Previously discussed Slack channels:\n"
                + "\n".join(f"  - {ch}" for ch in self.slack_channels)
            )
        if self.turn_summaries:
            recent = self.turn_summaries[-5:]  # Last 5 turns
            lines.append(
                "Recent conversation topics:\n"
                + "\n".join(f"  - {s}" for s in recent)
            )

        return "\n\n".join(lines) if lines else "(No prior context)"

    def has_context(self) -> bool:
        """Return True if any sources have been recorded."""
        return bool(
            self.repo_urls
            or self.drive_folders
            or self.slack_channels
            or self.turn_summaries
        )

    def to_dict(self) -> dict:
        """Serialize for logging/debugging."""
        return {
            "repo_urls": self.repo_urls,
            "drive_folders": self.drive_folders,
            "slack_channels": self.slack_channels,
            "turn_count": len(self.turn_summaries),
        }


# ──────────────────────────────────────────────────────────────
# Vector-aware tools (closure-based, for backward compatibility)
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Orchestrator with Plan-Action-Reflect loop + ConversationMemory
# ──────────────────────────────────────────────────────────────

class OrchestratorAgent:
    def __init__(self):
        """Initialize Orchestrator with multi-agent coordination and VectorDB"""
        self.logger = logging.getLogger("orchestrator")

        # Initialize Bedrock model
        self.bedrock_model = BedrockModel(
            model_id=os.getenv(
                "BEDROCK_MODEL_ID",
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
            )
        )

        # Initialize VectorDB singleton — orchestrator owns caching
        try:
            self.vector_store = get_vector_store()
            self.logger.info("Vector store initialized for orchestrator")
        except Exception as e:
            self.logger.warning(f"Vector store not available: {e}")
            self.vector_store = None

        # Conversation memory — persists across turns in the same session
        self.conversation_memory = ConversationMemory()

        # System prompt for final answer generation
        self.system_prompt = """
            You are Nowacoski, an intelligent onboarding assistant that creates
            personalized, efficient onboarding experiences for new employees.

            Your capabilities:
            1. **Google Drive Integration** - Fetch and analyze onboarding documents
            2. **GitHub Analysis** - Analyze codebases and repositories
            3. **Slack Integration** - Fetch and analyze team channel messages
            4. **Web Search** - Get current information via Tavily
            5. **Semantic Search** - Search all cached content by meaning
            6. **Intelligent Routing** - Use the right agent for each query

            Your goal is to help managers onboard new employees by:
            - Gathering context from Google Workspace, online resources, and code repos
            - Using cached data when available (faster and cheaper!)
            - Critically evaluating and prioritizing context
            - Generating a custom onboarding plan

            Output actionable plans that include:
            - Step-by-step onboarding roadmap
            - Key documents and tools to review
            - Suggested coding tasks or deliverables
            - Milestones to assess learning progress
            - Questions for the manager to clarify missing info

            ALWAYS use markdown formatting (headers, lists, tables) for clarity.
            Answer in the same language as the user's question.
        """

        # Create vector-aware tools via factory (backward compat)
        self._semantic_search_all, self._get_cache_status = _create_vector_tools(
            self.vector_store
        )

        # Legacy: keep tool list and messages for API compatibility
        tools = [
            trigger_tavily_agent,
            trigger_google_drive_agent,
            trigger_codebase_agent,
            trigger_slack_agent,
            self._semantic_search_all,
            self._get_cache_status,
        ]

        agent = Agent(
            tools=tools,
            model=self.bedrock_model,
            callback_handler=None,
        )
        self.messages = agent.messages
        self.tool_names = sorted(agent.tool_names)

    # ──────────────────────────────────────────────────────────
    # Main entry point: Plan → Action → Reflect → Fuse → Answer
    # ──────────────────────────────────────────────────────────

    def __call__(self, message: str) -> AgentResult:
        """Process user message with Plan-Action-Reflect loop.

        Flow:
        0. UPDATE MEMORY - Extract sources from message, update context
        1. PLAN   - Decompose query into sub-tasks (with conversation context)
        2. ACTION - Execute each sub-task (with centralized cache checks)
        3. REFLECT - Evaluate completeness (with conversation context)
        4. FUSE   - Combine all results into unified context
        5. ANSWER - Generate final response with fused context
        6. RECORD - Store discovered sources for future turns
        """
        self.logger.info(f"Orchestrator received: {message[:100]}")

        # 0. UPDATE MEMORY: extract sources from user message
        self.conversation_memory.extract_sources_from_message(message)
        if self.conversation_memory.has_context():
            self.logger.info(
                f"Conversation memory: {self.conversation_memory.to_dict()}"
            )

        # 1. PLAN: decompose query into sub-tasks (with memory context)
        plan = self._plan(message)

        if not plan:
            # Simple query (greeting, etc.) — direct LLM response, no tools
            self.logger.info("No tool plan needed, generating direct response")
            self.conversation_memory.add_turn_summary(
                f"General question: {message[:60]}"
            )
            agent = Agent(
                model=self.bedrock_model,
                system_prompt=self.system_prompt,
                callback_handler=self._callback_handler,
            )
            return agent(message)

        self.logger.info(
            f"Plan generated: {len(plan)} tool group(s) — "
            f"{[a['tool'] for a in plan]}"
        )

        # 2. ACTION: execute plan with centralized cache checks
        memory = self._execute_plan(plan, user_query=message)
        total_results = sum(len(m["results"]) for m in memory)
        self.logger.info(f"Action phase collected {total_results} total results")

        # 3. REFLECT: check if we need more information (with memory context)
        additional_plan = self._reflect(message, memory)
        if additional_plan:
            self.logger.info(
                f"Reflection: running {len(additional_plan)} additional tool group(s)"
            )
            additional_memory = self._execute_plan(additional_plan, user_query=message)
            memory.extend(additional_memory)

        # 4. FUSE: combine all results with deduplication
        fused_context = self._fuse_results(memory)
        self.logger.info(f"Fused context: {len(fused_context)} chars from {len(memory)} queries")

        # 5. ANSWER: generate final response with full context
        result = self._generate_answer(message, fused_context)

        # 6. RECORD: store discovered sources for future turns
        self.conversation_memory.record_sources_from_results(memory)
        self.conversation_memory.add_turn_summary(
            f"Query: {message[:60]} → {len(memory)} sources"
        )

        return result

    # ──────────────────────────────────────────────────────────
    # PLAN: Query decomposition + tool selection (stateful)
    # ──────────────────────────────────────────────────────────

    def _plan(self, query: str) -> Optional[List[Dict]]:
        """Decompose user query into sub-tasks with tool selection.

        Uses LLM to analyze the query and produce a structured plan
        indicating which tools to use and what sub-queries to run.

        The planner receives conversation memory so it can route
        follow-up questions to the correct tools without needing
        the user to re-provide URLs.

        Returns:
            list of {"tool": str, "queries": list[str]} or None
        """
        # Build conversation context
        memory_context = self.conversation_memory.get_context_for_planner()

        plan_prompt = f"""You are a planning module for an onboarding assistant.
Analyze the user query and decide which tools to use.

## Available Tools
1. **semantic_search** - Search cached documents in VectorDB (onboarding docs, past analyses, cached web results). Use this when information might already be cached from prior interactions.
2. **web_search** - Search the internet for current information via Tavily. Use for best practices, current trends, external knowledge.
3. **drive** - Fetch and analyze a Google Drive folder. Use when a drive.google.com URL is available.
4. **codebase** - Analyze a GitHub repository. Use when a github.com URL is available.
5. **slack** - Fetch and analyze Slack channel messages. Use when a Slack channel ID is available.

## Conversation Context
{memory_context}

## Rules
- If the query contains an actual URL starting with "drive.google.com/drive/folders/" → MUST include "drive" tool with the full URL as query
- If the query contains an actual URL starting with "github.com/" → MUST include "codebase" tool with the full URL as query
- If the query mentions a Slack channel ID (C followed by digits/letters) → MUST include "slack" tool
- If the query references a previously discussed source (listed in Conversation Context above) without providing the URL explicitly → use the stored URL from conversation context. For example, if the user asks "tell me more about KDA" and a GitHub repo was previously discussed, include "codebase" with that repo's URL.
- If the query merely *mentions* "Google Drive", "GitHub", or "Slack" without referencing any known source → use "semantic_search" instead
- For factual or current-info questions → include "web_search"
- For follow-up questions about previously discussed content → include "semantic_search"
- You can and SHOULD use multiple tools when appropriate
- Decompose complex queries into 1-3 focused sub-queries per tool
- For simple greetings or small talk, return null (no tools needed)

## Output Format
Return a JSON array:
[
  {{"tool": "tool_name", "queries": ["query1", "query2"]}},
  ...
]
Or: null (if no tools needed)

Return ONLY the JSON, no other text.

## User Query
{query}"""

        try:
            planner = Agent(
                model=self.bedrock_model,
                callback_handler=None,
            )
            response = planner(plan_prompt)
            parsed = _extract_json(str(response))

            if parsed and isinstance(parsed, list):
                # Validate plan structure
                validated = []
                for item in parsed:
                    if isinstance(item, dict) and "tool" in item and "queries" in item:
                        validated.append({
                            "tool": item["tool"],
                            "queries": item["queries"]
                            if isinstance(item["queries"], list)
                            else [item["queries"]],
                        })
                return validated if validated else None

            return None

        except Exception as e:
            self.logger.error(f"Plan generation failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # ACTION: Execute plan with centralized cache checks
    # ──────────────────────────────────────────────────────────

    def _execute_plan(self, plan: List[Dict], user_query: str = None) -> List[Dict]:
        """Execute each planned action with centralized cache management.

        For each tool call:
        1. Check VectorDB cache first (orchestrator decides)
        2. If cache is sufficient, use it — skip the agent
        3. If cache is insufficient, call the agent (pure worker)
        4. Cache new results for future queries

        Args:
            plan: List of planned actions with tool names and queries.
            user_query: The user's original question.
        """
        memory = []

        for action in plan:
            tool_name = action["tool"]
            queries = action["queries"]

            for query in queries:
                self.logger.info(f"Executing [{tool_name}]: {query[:80]}")

                try:
                    if tool_name == "semantic_search":
                        results = raw_vectordb_search(query)

                    elif tool_name == "web_search":
                        # Centralized cache: check for similar cached searches
                        cached = self._check_cache(
                            query, folder_id="tavily_searches", threshold=0.7
                        )
                        if cached:
                            self.logger.info(
                                "Using cached web search (orchestrator decision)"
                            )
                            results = [
                                {
                                    "title": "Cached Web Search",
                                    "url": "",
                                    "content": cached,
                                }
                            ]
                        else:
                            results = raw_web_search(query)
                            # Cache results for future queries
                            if results:
                                self._cache_results(
                                    tool_name, query, results,
                                    folder_id="tavily_searches",
                                )

                    elif tool_name == "drive":
                        # Resolve the actual Drive URL
                        drive_url = self._resolve_drive_url(query)
                        if not drive_url:
                            self.logger.warning(
                                f"No Drive URL found for: {query[:60]}"
                            )
                            results = [
                                {
                                    "title": "Drive Content",
                                    "url": "",
                                    "content": (
                                        "Could not determine Drive folder URL. "
                                        "Please provide a Google Drive link."
                                    ),
                                }
                            ]
                        else:
                            folder_id = self._extract_drive_folder_id(drive_url)
                            cached = self._check_cache(
                                user_query or query,
                                folder_id=folder_id,
                                threshold=0.5,
                            ) if folder_id else None

                            if cached:
                                self.logger.info(
                                    "Using cached drive content "
                                    "(orchestrator decision)"
                                )
                                results = [
                                    {
                                        "title": "Cached Drive Content",
                                        "url": drive_url,
                                        "content": cached,
                                    }
                                ]
                            else:
                                drive_response = trigger_google_drive_agent(
                                    drive_url
                                )
                                results = [
                                    {
                                        "title": "Google Drive Content",
                                        "url": drive_url,
                                        "content": str(drive_response),
                                    }
                                ]
                                # Cache for future queries
                                if results and folder_id:
                                    self._cache_results(
                                        tool_name, query, results,
                                        folder_id=folder_id,
                                    )

                    elif tool_name == "codebase":
                        # Resolve the actual GitHub URL.
                        # The planner may pass a descriptive query like
                        # "KDA attention implementation" instead of the URL.
                        repo_url = self._resolve_repo_url(query)
                        if not repo_url:
                            self.logger.warning(
                                f"No GitHub URL found in query or memory "
                                f"for codebase tool: {query[:60]}"
                            )
                            results = [
                                {
                                    "title": "Codebase Analysis",
                                    "url": "",
                                    "content": (
                                        "Could not determine repository URL. "
                                        "Please provide a GitHub URL."
                                    ),
                                }
                            ]
                        else:
                            # Use the planner's descriptive query as the
                            # specific question (e.g. "CLI tool implementation
                            # and entry points") — it's cleaner than the full
                            # user message which may contain URLs, colons, etc.
                            # that break GitHub code search.
                            if query and repo_url != query:
                                # Planner gave a descriptive sub-query
                                specific_query = query
                            elif user_query and repo_url != user_query:
                                # Fallback to user message (but not if it IS
                                # the URL)
                                specific_query = user_query
                            else:
                                specific_query = None

                            repo_id = self._extract_repo_id(repo_url)
                            cache_folder = (
                                f"github-{repo_id}" if repo_id else None
                            )

                            cached_content = None
                            cache_relevant = False

                            if cache_folder:
                                cached_content = self._check_cache(
                                    user_query or query,
                                    folder_id=cache_folder,
                                    threshold=0.5,
                                )
                                if cached_content:
                                    cache_relevant = True
                                    self.logger.info(
                                        "Using cached codebase analysis "
                                        "(orchestrator decision)"
                                    )

                            if cache_relevant:
                                results = [
                                    {
                                        "title": "Cached Codebase Analysis",
                                        "url": repo_url,
                                        "content": cached_content,
                                    }
                                ]
                            else:
                                # Call agent as pure worker
                                self.logger.info(
                                    f"Calling codebase agent: "
                                    f"url={repo_url}, query={specific_query}"
                                )
                                code_response = trigger_codebase_agent(
                                    repo_url, query=specific_query
                                )
                                results = [
                                    {
                                        "title": "Codebase Analysis",
                                        "url": repo_url,
                                        "content": str(code_response),
                                    }
                                ]
                                # Cache for future queries
                                if results and cache_folder:
                                    self._cache_results(
                                        tool_name, query, results,
                                        folder_id=cache_folder,
                                        metadata={
                                            "repo_url": repo_url,
                                            "repo_id": repo_id,
                                        },
                                    )

                    elif tool_name == "slack":
                        # Resolve the actual Slack channel ID
                        channel_id = self._resolve_slack_channel(query)
                        if not channel_id:
                            self.logger.warning(
                                f"No Slack channel ID found for: {query[:60]}"
                            )
                            results = [
                                {
                                    "title": "Slack Content",
                                    "url": "",
                                    "content": (
                                        "Could not determine Slack channel ID. "
                                        "Please provide a channel ID."
                                    ),
                                }
                            ]
                        else:
                            slack_folder = f"slack-{channel_id}"
                            cached = self._check_cache(
                                user_query or query,
                                folder_id=slack_folder,
                                threshold=0.5,
                            )
                            if cached:
                                self.logger.info(
                                    "Using cached slack content "
                                    "(orchestrator decision)"
                                )
                                results = [
                                    {
                                        "title": f"Cached Slack #{channel_id}",
                                        "url": "",
                                        "content": cached,
                                    }
                                ]
                            else:
                                results = raw_slack_search(channel_id)
                                # Cache for future queries
                                if results:
                                    self._cache_results(
                                        tool_name, query, results,
                                        folder_id=slack_folder,
                                        metadata={
                                            "channel_id": channel_id,
                                        },
                                    )

                    else:
                        self.logger.warning(f"Unknown tool: {tool_name}")
                        results = []

                    memory.append({
                        "query": query,
                        "source": tool_name,
                        "results": results,
                    })

                    self.logger.info(
                        f"[{tool_name}] returned {len(results)} results for: {query[:50]}"
                    )

                except Exception as e:
                    self.logger.error(f"[{tool_name}] failed for '{query[:50]}': {e}")
                    memory.append({
                        "query": query,
                        "source": tool_name,
                        "results": [],
                    })

        return memory

    # ──────────────────────────────────────────────────────────
    # Centralized cache helpers
    # ──────────────────────────────────────────────────────────

    def _check_cache(
        self, query: str, folder_id: str = None, threshold: float = 0.5
    ) -> Optional[str]:
        """Check VectorDB cache for relevant content.

        Returns cached content string if similarity >= threshold, else None.
        This centralizes all cache-vs-fetch decisions in the orchestrator.

        Uses ``use_query_embedding=True`` so the search compares the
        incoming short query against the stored short query embedding
        (not the full document embedding).  This avoids the "embedding
        dilution" problem where a 5-word query gets low similarity
        against a 3000-char document embedding.
        """
        if self.vector_store is None or self.vector_store.collection is None:
            return None

        try:
            results = self.vector_store.search_similar(
                query=query,
                folder_id=folder_id,
                top_k=1,
                threshold=threshold,
                use_query_embedding=True,
            )
            if results:
                similarity = results[0].get("similarity", 0)
                self.logger.info(
                    f"Cache HIT: {similarity:.0%} relevance "
                    f"(threshold: {threshold:.0%}, folder: {folder_id})"
                )
                return results[0].get("content", "")
            return None
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
            return None

    def _cache_results(
        self,
        tool_name: str,
        query: str,
        results: List[Dict],
        folder_id: str = None,
        metadata: Dict = None,
    ) -> None:
        """Cache tool results in VectorDB for future queries.

        Called by the orchestrator after a fresh fetch from any agent.
        Stores a separate ``query_embedding`` (from the short query string)
        alongside the full document embedding so that future cache lookups
        compare short-to-short and avoid embedding dilution.
        """
        if not self.vector_store:
            return

        try:
            combined = "\n\n".join(
                f"[{r.get('title', '')}] {r.get('content', '')}"
                for r in results
                if r.get("content", "").strip()
            )
            if not combined.strip():
                return

            meta = {
                "type": f"{tool_name}_result",
                "query": query,
                **(metadata or {}),
            }

            doc_id = self.vector_store.store_document(
                content=f"Query: {query}\n\nResult:\n{combined}",
                source=f"{tool_name}: {query[:50]}",
                folder_id=folder_id or f"{tool_name}_results",
                metadata=meta,
                query_text=query,  # separate short-query embedding for cache lookups
            )
            if doc_id:
                self.logger.info(f"Cached {tool_name} result: {doc_id}")
        except Exception as e:
            self.logger.warning(f"Could not cache {tool_name} result: {e}")

    @staticmethod
    def _extract_repo_id(query: str) -> Optional[str]:
        """Extract repo ID (owner-name) from a GitHub URL or query."""
        match = re.search(r"github\.com/([\w\-]+/[\w\-]+)", query)
        if match:
            return match.group(1).lower().replace("/", "-")
        return None

    @staticmethod
    def _extract_drive_folder_id(query: str) -> Optional[str]:
        """Extract folder ID from a Google Drive URL."""
        if "/folders/" in query:
            return query.split("/folders/")[-1].split("?")[0].split("#")[0]
        return None

    def _resolve_repo_url(self, query: str) -> Optional[str]:
        """Resolve a GitHub repository URL from the query or conversation memory.

        The planner may produce descriptive queries (e.g. "KDA attention
        implementation") instead of the actual URL.  This method:
        1. Checks if the query itself contains a github.com URL.
        2. Checks the original user message for a github.com URL.
        3. Falls back to the most recently discussed repo in conversation memory.
        """
        # 1. Query itself contains a URL
        match = re.search(r"https?://github\.com/[\w\-]+/[\w\-]+", query)
        if match:
            return match.group(0).rstrip("/")

        # 2. Check conversation memory (most recent repo first)
        if self.conversation_memory.repo_urls:
            url = self.conversation_memory.repo_urls[-1]
            self.logger.info(f"Resolved repo URL from conversation memory: {url}")
            return url

        return None

    def _resolve_drive_url(self, query: str) -> Optional[str]:
        """Resolve a Google Drive folder URL from the query or conversation memory."""
        match = re.search(
            r"https?://drive\.google\.com/drive/folders/[\w\-]+", query
        )
        if match:
            return match.group(0)

        if self.conversation_memory.drive_folders:
            url = self.conversation_memory.drive_folders[-1]
            self.logger.info(f"Resolved drive URL from conversation memory: {url}")
            return url

        return None

    def _resolve_slack_channel(self, query: str) -> Optional[str]:
        """Resolve a Slack channel ID from the query or conversation memory."""
        match = re.search(r"\b(C[A-Z0-9]{8,})\b", query)
        if match:
            return match.group(1)

        if self.conversation_memory.slack_channels:
            ch = self.conversation_memory.slack_channels[-1]
            self.logger.info(f"Resolved channel from conversation memory: {ch}")
            return ch

        return None

    # ──────────────────────────────────────────────────────────
    # REFLECT: Evaluate completeness (with conversation memory)
    # ──────────────────────────────────────────────────────────

    def _reflect(
        self, original_query: str, memory: List[Dict]
    ) -> Optional[List[Dict]]:
        """Evaluate whether collected results are sufficient.

        If more information is needed, returns an additional plan
        (same format as _plan output). Otherwise returns None.

        Receives conversation memory context so it can suggest
        re-fetching from previously discussed sources.
        """
        # Build summary of collected results
        memory_items = []
        for m in memory:
            item = {
                "query": m["query"],
                "source": m["source"],
                "result_count": len(m["results"]),
                "has_content": any(
                    bool(r.get("content", "").strip()) for r in m["results"]
                ),
            }
            # Extract actionable metadata
            for r in m["results"]:
                meta = r.get("metadata", {})
                if meta.get("repo_url"):
                    item["repo_url"] = meta["repo_url"]
                    item["similarity"] = r.get("similarity", 0.0)
                if meta.get("channel_id"):
                    item["channel_id"] = meta["channel_id"]
            memory_items.append(item)

        memory_summary = json.dumps(memory_items, indent=2)
        conv_context = self.conversation_memory.get_context_for_planner()

        reflect_prompt = f"""You are a reflection module for an onboarding assistant.
Evaluate whether the collected information is sufficient to answer the user's query.

## Original Query
{original_query}

## Information Collected So Far
{memory_summary}

## Conversation Context (previously discussed sources)
{conv_context}

## Available Tools
- semantic_search: Search cached VectorDB documents
- web_search: Search the internet via Tavily
- codebase: Analyze a GitHub repository. Use a repo URL from conversation context if available.
- drive: Fetch a Google Drive folder. Use a drive URL from conversation context if available.
- slack: Fetch Slack channel messages. Use a channel ID from conversation context if available.

## Instructions
- If the information is SUFFICIENT to provide a good answer, return: null
- If MORE information is needed, return a JSON plan with at most 3 additional queries:
[
  {{"tool": "tool_name", "queries": ["additional_query_1"]}},
  ...
]
- Only suggest additional queries that would meaningfully improve the answer
- Do NOT repeat queries that were already executed
- Prefer web_search for general knowledge gaps
- Use codebase/drive/slack with URLs from conversation context for source-specific depth

Return ONLY the JSON or null, no other text."""

        try:
            reflector = Agent(
                model=self.bedrock_model,
                callback_handler=None,
            )
            response = reflector(reflect_prompt)
            parsed = _extract_json(str(response))

            if parsed and isinstance(parsed, list):
                validated = []
                for item in parsed:
                    if isinstance(item, dict) and "tool" in item and "queries" in item:
                        validated.append({
                            "tool": item["tool"],
                            "queries": item["queries"]
                            if isinstance(item["queries"], list)
                            else [item["queries"]],
                        })
                return validated if validated else None

            return None

        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # FUSE: Combine all results into unified context
    # ──────────────────────────────────────────────────────────

    def _fuse_results(self, memory: List[Dict]) -> str:
        """Fuse all collected results into a unified, deduplicated context string.

        Combines RAG results + web search results + drive/codebase content
        into a single numbered reference list (Level-2 fusion).
        Deduplication is based on content hash of first 200 characters.
        """
        seen_content = set()
        fused_items = []
        idx = 1

        for item in memory:
            source_tag = item["source"]

            for result in item["results"]:
                content = result.get("content", "").strip()
                if not content:
                    continue

                # Deduplicate by hashing first 200 chars
                content_hash = hash(content[:200])
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)

                title = result.get("title", "")
                url = result.get("url", "")

                header = f"[{source_tag}] {title}"
                if url:
                    header += f" ({url})"

                fused_items.append(f"{idx}. {header}\n{content}")
                idx += 1

        if not fused_items:
            return "(No reference materials found)"

        return "\n\n".join(fused_items)

    # ──────────────────────────────────────────────────────────
    # ANSWER: Generate final response with fused context
    # ──────────────────────────────────────────────────────────

    def _generate_answer(self, query: str, fused_context: str) -> AgentResult:
        """Generate final answer using all fused reference materials.

        Uses a Strands Agent (no tools) so the return type is AgentResult,
        keeping compatibility with the Streamlit UI.
        """
        final_prompt = f"""Based on the following reference materials collected from
multiple sources (knowledge base, web search, documents, codebases),
provide a comprehensive and well-structured answer to the user's question.

## Reference Materials
{fused_context}

## User Question
{query}

## Instructions
- Synthesize information from ALL provided sources
- Use markdown formatting: headers (####), bullet lists, tables, code blocks
- Be specific and actionable
- If creating an onboarding plan, include a timeline and milestones
- If reference materials are empty or insufficient, use your own knowledge
- Answer in the same language as the user's question
- NEVER start with a heading — begin with a direct response"""

        agent = Agent(
            model=self.bedrock_model,
            system_prompt=self.system_prompt,
            callback_handler=self._callback_handler,
        )
        return agent(final_prompt)

    # ──────────────────────────────────────────────────────────
    # Utility methods
    # ──────────────────────────────────────────────────────────

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
        """Clear cache for specific folder or all caches."""
        if not self.vector_store:
            return "Vector store not available"

        if folder_id:
            deleted = self.vector_store.delete_folder_documents(folder_id)
            return f"Cleared {deleted} documents from folder {folder_id}"
        else:
            return (
                "Please specify a folder_id to clear. "
                "Options: 'github-repo-name', 'tavily_searches', "
                "or your Drive folder ID"
            )


if __name__ == "__main__":
    # Quick smoke test
    agent = OrchestratorAgent()
    print("Orchestrator initialized with Plan-Action-Reflect loop enabled")
    print(f"Available tools: {agent.tool_names}")
