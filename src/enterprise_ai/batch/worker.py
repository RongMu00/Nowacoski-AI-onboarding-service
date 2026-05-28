"""Temporal worker entry point for batch onboarding.

Usage:
    python -m enterprise_ai.batch.worker

Connects to the Temporal server and starts processing onboarding
workflows and activities.
"""

import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from enterprise_ai.batch.activities import run_onboarding_activity
from enterprise_ai.batch.workflows import (
    BatchOnboardingWorkflow,
    SingleHireOnboardingWorkflow,
)

logger = logging.getLogger("batch_worker")

TASK_QUEUE = "onboarding-queue"


async def run_worker() -> None:
    """Connect to Temporal and run the worker until interrupted."""
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    logger.info(f"Connecting to Temporal at {temporal_host}")

    client = await Client.connect(temporal_host)
    logger.info("Connected to Temporal – starting worker")

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BatchOnboardingWorkflow, SingleHireOnboardingWorkflow],
        activities=[run_onboarding_activity],
        max_concurrent_activities=3,
    )

    await worker.run()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker shut down")


if __name__ == "__main__":
    main()
