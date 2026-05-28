import logging
import os
from typing import Dict, List

from strands import Agent
from strands.models import BedrockModel

logger = logging.getLogger("slack_agent")


class SlackAgent:
    """Pure worker agent for Slack channel message fetching and analysis.

    This agent only fetches from Slack and generates summaries — it does NOT
    manage its own cache.  All cache-vs-fetch decisions are made by the
    orchestrator.

    Requires SLACK_BOT_TOKEN environment variable (Bot User OAuth Token
    with channels:history, channels:read, users:read scopes).
    """

    def __init__(self):
        self.logger = logging.getLogger("slack_agent")

        from dotenv import load_dotenv
        load_dotenv()

        self.slack_token = os.getenv("SLACK_BOT_TOKEN", "")

        # Initialize Slack client
        self.client = None
        self.is_authenticated = False

        if self.slack_token:
            try:
                from slack_sdk import WebClient
                self.client = WebClient(token=self.slack_token)
                auth_response = self.client.auth_test()
                self.is_authenticated = True
                self.logger.info(
                    f"Slack authenticated as: {auth_response['user']}"
                )
            except Exception as e:
                self.logger.error(f"Slack authentication failed: {e}")
        else:
            self.logger.warning("SLACK_BOT_TOKEN not set")

        # Initialize Bedrock model for summarization
        self.bedrock_model = BedrockModel(
            model_id=os.getenv("BEDROCK_MODEL_ID")
        )

        self.system_prompt = """
            You are the Slack Agent, a focused internal communication analyst
            in the Nowacoski onboarding system.

            Your role is to analyze Slack channel messages and extract key
            information relevant to onboarding a new employee.

            From the provided messages, identify:
            - Team communication patterns and norms
            - Key projects and initiatives being discussed
            - Important announcements or decisions
            - Recurring topics or pain points
            - Team members and their roles/expertise

            Return a structured, concise summary that helps the orchestrator
            build a better onboarding plan.
        """

    def __call__(self, channel_id: str, limit: int = 50) -> str:
        """Fetch and analyze Slack channel messages.

        Always fetches from Slack API — the orchestrator decides whether to
        call this method or use cached content.

        Args:
            channel_id: Slack channel ID (e.g., "C0123456789")
            limit: Maximum number of messages to fetch

        Returns:
            LLM-generated summary of channel content
        """
        self.logger.info(f"Slack agent processing channel: {channel_id}")

        messages = self._fetch_channel_messages(channel_id, limit)
        if not messages:
            return f"No messages found in channel {channel_id} (or access denied)"

        combined_text = self._format_messages(messages)
        return self._summarize(channel_id, combined_text)

    def search_raw(self, channel_id: str, limit: int = 50) -> List[Dict]:
        """Return raw Slack messages as structured data for fusion.

        Unlike __call__() which generates an LLM summary, this method
        returns structured message data suitable for result fusion
        in the Plan-Action-Reflect loop.

        Returns:
            list of {"title": str, "url": str, "content": str}
        """
        self.logger.info(f"Slack raw search for channel: {channel_id}")

        messages = self._fetch_channel_messages(channel_id, limit)
        if not messages:
            return []

        snippets = []
        for msg in messages:
            user = msg.get("user_name", msg.get("user", "unknown"))
            text = msg.get("text", "")

            if text.strip():
                snippets.append({
                    "title": f"Slack message by {user}",
                    "url": "",
                    "content": f"[{user}] {text}",
                })

        self.logger.info(f"Slack raw search returned {len(snippets)} messages")
        return snippets

    # ──────────────────────────────────────────────────────────
    # Slack API methods
    # ──────────────────────────────────────────────────────────

    def _fetch_channel_messages(
        self, channel_id: str, limit: int = 50
    ) -> List[Dict]:
        """Fetch recent messages from a Slack channel."""
        if not self.is_authenticated or not self.client:
            self.logger.warning("Slack client not authenticated")
            return []

        try:
            # Fetch conversation history
            result = self.client.conversations_history(
                channel=channel_id,
                limit=limit,
            )

            messages = result.get("messages", [])
            self.logger.info(
                f"Fetched {len(messages)} messages from channel {channel_id}"
            )

            # Enrich messages with user display names
            enriched = []
            user_cache = {}

            for msg in messages:
                user_id = msg.get("user", "")
                if user_id and user_id not in user_cache:
                    try:
                        user_info = self.client.users_info(user=user_id)
                        user_cache[user_id] = (
                            user_info["user"]["profile"].get("display_name")
                            or user_info["user"]["real_name"]
                        )
                    except Exception:
                        user_cache[user_id] = user_id

                msg["user_name"] = user_cache.get(user_id, user_id)
                enriched.append(msg)

            return enriched

        except Exception as e:
            self.logger.error(f"Error fetching Slack messages: {e}")
            return []

    def _get_channel_name(self, channel_id: str) -> str:
        """Get channel name from ID."""
        if not self.client:
            return channel_id

        try:
            info = self.client.conversations_info(channel=channel_id)
            return info["channel"].get("name", channel_id)
        except Exception:
            return channel_id

    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages into readable text for caching and LLM input."""
        lines = []
        for msg in reversed(messages):  # Chronological order
            user = msg.get("user_name", msg.get("user", "unknown"))
            text = msg.get("text", "")
            if text.strip():
                lines.append(f"[{user}]: {text}")
        return "\n".join(lines)

    def _summarize(self, channel_id: str, content: str) -> str:
        """Generate LLM summary of channel messages."""
        prompt = f"""Analyze these Slack messages from channel {channel_id}
for onboarding purposes:

{content}

Provide a structured summary covering:
1. Key topics discussed
2. Active team members and their roles
3. Important decisions or announcements
4. Relevant projects or initiatives
5. Team norms and communication patterns
"""
        try:
            agent = Agent(
                model=self.bedrock_model,
                system_prompt=self.system_prompt,
            )
            response = agent(prompt)
            return str(response)
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return f"Fetched {content.count(chr(10)) + 1} messages but summarization failed: {e}"

