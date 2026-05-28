"""Temporal activities for batch onboarding.

Each activity wraps the existing OrchestratorAgent — no orchestrator
changes needed.  A fresh agent is created per hire so that
ConversationMemory is isolated while the MongoVectorStore singleton
(expensive connection + embedding model) is shared across invocations.
"""

import asyncio
import logging

from temporalio import activity

from enterprise_ai.batch.models import NewHire

logger = logging.getLogger("batch_activities")


def _build_onboarding_prompt(hire: NewHire) -> str:
    """Build a natural-language prompt from a NewHire record.

    The prompt mimics what a manager would type into the chat UI so the
    existing orchestrator's planner routes it correctly.
    """
    parts = [
        f"Create a personalized onboarding plan for {hire.name}, "
        f"who is joining as a {hire.role}.",
    ]

    if hire.github_repo_url:
        parts.append(
            f"Analyze this GitHub repository they'll be working on: "
            f"{hire.github_repo_url}"
        )
    if hire.drive_folder_url:
        parts.append(
            f"Review the onboarding documents in this Google Drive folder: "
            f"{hire.drive_folder_url}"
        )
    if hire.slack_channel_id:
        parts.append(
            f"Check the recent messages in Slack channel {hire.slack_channel_id} "
            f"for team context."
        )

    parts.append(
        "Include a detailed week-by-week onboarding roadmap, "
        "key resources to review, suggested first tasks, "
        "and milestones to track progress."
    )

    return " ".join(parts)


@activity.defn
async def run_onboarding_activity(hire_dict: dict) -> dict:
    """Execute the full Plan-Action-Reflect loop for one new hire.

    Receives/returns plain dicts (Temporal serialises them as JSON).
    The synchronous orchestrator runs in a thread-pool executor to
    avoid blocking the Temporal activity event loop.
    """
    hire = NewHire(**hire_dict)
    logger.info(f"Starting onboarding for: {hire.name} ({hire.role})")

    prompt = _build_onboarding_prompt(hire)

    loop = asyncio.get_event_loop()

    def _run_sync() -> str:
        # Import here so the module can be imported even when the
        # orchestrator's heavy dependencies aren't installed yet
        # (e.g. during workflow-only unit tests).
        from enterprise_ai.agents.orchestrator_caching import OrchestratorAgent

        agent = OrchestratorAgent()
        result = agent(prompt)
        return str(result)

    try:
        plan_text = await loop.run_in_executor(None, _run_sync)
        logger.info(f"Completed onboarding for: {hire.name}")
        return {
            "name": hire.name,
            "role": hire.role,
            "status": "completed",
            "onboarding_plan": plan_text,
        }
    except Exception as e:
        logger.error(f"Failed onboarding for {hire.name}: {e}")
        raise  # Let Temporal's retry policy handle it
