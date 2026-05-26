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
# Orchestrator with Plan-Action-Reflect loop
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

        # Initialize VectorDB singleton for orchestration memory
        try:
            self.vector_store = get_vector_store()
            self.logger.info("Vector store initialized for orchestrator")
        except Exception as e:
            self.logger.warning(f"Vector store not available: {e}")
            self.vector_store = None

        # Initialize sub-agents (lazy, kept for backward compat)
        self.drive_agent = None
        self.tavily_agent = None
        self.codebase_agent = None

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
        1. PLAN   - Decompose query into sub-tasks with tool selection
        2. ACTION - Execute each sub-task, collect structured results
        3. REFLECT - Evaluate completeness, optionally run more queries
        4. FUSE   - Combine all results into unified context
        5. ANSWER - Generate final response with fused context
        """
        self.logger.info(f"Orchestrator received: {message[:100]}")

        # 1. PLAN: decompose query into sub-tasks
        plan = self._plan(message)

        if not plan:
            # Simple query (greeting, etc.) — direct LLM response, no tools
            self.logger.info("No tool plan needed, generating direct response")
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

        # 2. ACTION: execute plan, collect structured results
        memory = self._execute_plan(plan, user_query=message)
        total_results = sum(len(m["results"]) for m in memory)
        self.logger.info(f"Action phase collected {total_results} total results")

        # 3. REFLECT: check if we need more information
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
        return self._generate_answer(message, fused_context)

    # ──────────────────────────────────────────────────────────
    # PLAN: Query decomposition + tool selection
    # ──────────────────────────────────────────────────────────

    def _plan(self, query: str) -> Optional[List[Dict]]:
        """Decompose user query into sub-tasks with tool selection.

        Uses LLM to analyze the query and produce a structured plan
        indicating which tools to use and what sub-queries to run.

        Returns:
            list of {"tool": str, "queries": list[str]} or None
        """
        plan_prompt = f"""You are a planning module for an onboarding assistant.
Analyze the user query and decide which tools to use.

## Available Tools
1. **semantic_search** - Search cached documents in VectorDB (onboarding docs, past analyses, cached web results). Use this when information might already be cached from prior interactions.
2. **web_search** - Search the internet for current information via Tavily. Use for best practices, current trends, external knowledge.
3. **drive** - Fetch and analyze a Google Drive folder. Use ONLY when the user provides a drive.google.com link.
4. **codebase** - Analyze a GitHub repository. Use ONLY when the user provides a github.com link.
5. **slack** - Fetch and analyze Slack channel messages. Use ONLY when the user provides a Slack channel ID (e.g., "C0123456789") or mentions Slack.

## Rules
- If the query contains an actual URL starting with "drive.google.com/drive/folders/" → MUST include "drive" tool with the full URL as query
- If the query merely *mentions* "Google Drive" or "drive" without an actual drive.google.com URL → do NOT use "drive" tool. Use "semantic_search" instead to look up previously cached Drive content.
- If the query contains an actual URL starting with "github.com/" → MUST include "codebase" tool with the full URL as query
- If the query merely *mentions* "GitHub" or a repository without an actual github.com URL → do NOT use "codebase" tool. Use "semantic_search" instead.
- If the query mentions a Slack channel ID (C followed by digits/letters) → MUST include "slack" tool
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
    # ACTION: Execute plan and collect structured results
    # ──────────────────────────────────────────────────────────

    def _execute_plan(self, plan: List[Dict], user_query: str = None) -> List[Dict]:
        """Execute each planned action and collect structured results.

        Each tool returns results in a uniform format:
        {"query": str, "source": str, "results": list[{"title", "url", "content"}]}

        This enables fusion across heterogeneous sources.

        Args:
            plan: List of planned actions with tool names and queries.
            user_query: The user's original question. Passed to codebase agent
                       for hybrid cache + targeted file fetching.
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
                        results = raw_web_search(query)

                    elif tool_name == "drive":
                        # Drive agent returns a formatted plan string;
                        # wrap as structured data for fusion
                        drive_response = trigger_google_drive_agent(query)
                        results = [
                            {
                                "title": "Google Drive Content",
                                "url": query,
                                "content": str(drive_response),
                            }
                        ]

                    elif tool_name == "codebase":
                        # Codebase agent returns analysis string;
                        # wrap as structured data for fusion.
                        # Pass user_query for hybrid cache + targeted fetch.
                        code_response = trigger_codebase_agent(
                            query, query=user_query
                        )
                        results = [
                            {
                                "title": "Codebase Analysis",
                                "url": query,
                                "content": str(code_response),
                            }
                        ]

                    elif tool_name == "slack":
                        # Slack agent returns structured messages
                        results = raw_slack_search(query)

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
    # REFLECT: Evaluate completeness, optionally plan more
    # ──────────────────────────────────────────────────────────

    def _reflect(
        self, original_query: str, memory: List[Dict]
    ) -> Optional[List[Dict]]:
        """Evaluate whether collected results are sufficient.

        If more information is needed, returns an additional plan
        (same format as _plan output). Otherwise returns None.
        """
        # Build a summary of what we've collected so far
        memory_summary = json.dumps(
            [
                {
                    "query": m["query"],
                    "source": m["source"],
                    "result_count": len(m["results"]),
                    "has_content": any(
                        bool(r.get("content", "").strip()) for r in m["results"]
                    ),
                }
                for m in memory
            ],
            indent=2,
        )

        reflect_prompt = f"""You are a reflection module for an onboarding assistant.
Evaluate whether the collected information is sufficient to answer the user's query.

## Original Query
{original_query}

## Information Collected So Far
{memory_summary}

## Available Tools
- semantic_search: Search cached VectorDB documents (use for follow-up questions about previously fetched content)
- web_search: Search the internet via Tavily
- drive: Fetch Google Drive folder (ONLY if the original query contains a drive.google.com URL — never for follow-up questions)
- codebase: Analyze GitHub repo (ONLY if the original query contains a github.com URL — never for follow-up questions)
- slack: Fetch Slack channel messages (only if user provided a Slack channel ID)

## Instructions
- If the information is SUFFICIENT to provide a good answer, return: null
- If MORE information is needed, return a JSON plan with at most 3 additional queries:
[
  {{"tool": "tool_name", "queries": ["additional_query_1"]}},
  ...
]
- Only suggest additional queries that would meaningfully improve the answer
- Do NOT repeat queries that were already executed
- Prefer web_search for filling knowledge gaps

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
