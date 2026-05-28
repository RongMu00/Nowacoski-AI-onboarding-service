"""Temporal workflows for batch onboarding.

BatchOnboardingWorkflow starts one SingleHireOnboardingWorkflow per hire
as child workflows.  Progress is exposed via @workflow.query so the
Streamlit UI can poll without blocking.
"""

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from enterprise_ai.batch.models import NewHire


# ── Single hire ──────────────────────────────────────────────────────


@workflow.defn
class SingleHireOnboardingWorkflow:
    """Run the full Plan-Action-Reflect loop for one new hire."""

    def __init__(self) -> None:
        self._status: str = "pending"
        self._result: dict[str, Any] | None = None

    @workflow.run
    async def run(self, hire_dict: dict) -> dict:
        self._status = "in_progress"
        try:
            self._result = await workflow.execute_activity(
                "run_onboarding_activity",
                hire_dict,
                start_to_close_timeout=timedelta(minutes=20),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=30),
                    backoff_coefficient=2.0,
                ),
            )
            self._status = "completed"
            return self._result
        except Exception as exc:
            self._status = "failed"
            self._result = {
                "name": hire_dict.get("name", "unknown"),
                "role": hire_dict.get("role", "unknown"),
                "status": "failed",
                "error": str(exc),
            }
            raise

    @workflow.query
    def get_status(self) -> str:
        return self._status

    @workflow.query
    def get_result(self) -> dict[str, Any] | None:
        return self._result


# ── Batch (parent) ───────────────────────────────────────────────────


@workflow.defn
class BatchOnboardingWorkflow:
    """Orchestrate onboarding for a list of new hires.

    Starts one child workflow per hire and tracks aggregate progress.
    """

    def __init__(self) -> None:
        self._total: int = 0
        self._completed: int = 0
        self._failed: int = 0
        self._results: list[dict[str, Any]] = []

    @workflow.run
    async def run(self, hires: list[dict]) -> list[dict]:
        self._total = len(hires)

        # Start all child workflows concurrently.
        # Worker's max_concurrent_activities limits actual parallelism.
        handles = []
        for idx, hire_dict in enumerate(hires):
            child_id = (
                f"{workflow.info().workflow_id}"
                f"/hire-{idx}-{hire_dict.get('name', 'unknown')}"
            )
            handle = await workflow.start_child_workflow(
                SingleHireOnboardingWorkflow.run,
                hire_dict,
                id=child_id,
            )
            handles.append((hire_dict, handle))

        # Wait for every child; collect results regardless of success/failure.
        for hire_dict, handle in handles:
            try:
                result = await handle
                self._completed += 1
                self._results.append(result)
            except Exception as exc:
                self._failed += 1
                self._results.append(
                    {
                        "name": hire_dict.get("name", "unknown"),
                        "role": hire_dict.get("role", "unknown"),
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        return self._results

    @workflow.query
    def get_progress(self) -> dict[str, int]:
        return {
            "total": self._total,
            "completed": self._completed,
            "failed": self._failed,
            "in_progress": self._total - self._completed - self._failed,
        }

    @workflow.query
    def get_results(self) -> list[dict[str, Any]]:
        return list(self._results)
