"""Streamlit UI components for Temporal batch onboarding.

Provides three render functions that `ui.py` calls when the user
selects Batch mode:

- render_batch_upload()   – CSV upload, preview table, start button
- render_batch_progress() – live progress bar + per-hire results
"""

import asyncio
import logging
import uuid
from dataclasses import asdict

import streamlit as st
from temporalio.client import Client

from enterprise_ai.batch.models import parse_csv
from enterprise_ai.batch.worker import TASK_QUEUE
from enterprise_ai.batch.workflows import BatchOnboardingWorkflow

logger = logging.getLogger("batch_ui")


# ── Helpers ──────────────────────────────────────────────────────────

def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return the running loop, or create one if none exists."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _run_async(coro):
    """Run a coroutine from synchronous Streamlit code."""
    loop = _get_or_create_event_loop()
    if loop.is_running():
        # Streamlit ≥ 1.27 may already have a loop; create a new one
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return loop.run_until_complete(coro)


async def _get_temporal_client() -> Client:
    import os
    host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    return await Client.connect(host)


# ── Start workflow ───────────────────────────────────────────────────

async def _start_batch_workflow(hires_dicts: list[dict]) -> str:
    """Start a BatchOnboardingWorkflow and return its workflow ID."""
    client = await _get_temporal_client()
    workflow_id = f"batch-onboard-{uuid.uuid4().hex[:8]}"
    await client.start_workflow(
        BatchOnboardingWorkflow.run,
        hires_dicts,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    return workflow_id


# ── Query workflow ───────────────────────────────────────────────────

async def _query_progress(workflow_id: str) -> dict:
    client = await _get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    return await handle.query(BatchOnboardingWorkflow.get_progress)


async def _query_results(workflow_id: str) -> list[dict]:
    client = await _get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    return await handle.query(BatchOnboardingWorkflow.get_results)


# ── UI components ────────────────────────────────────────────────────

def render_batch_upload() -> None:
    """CSV upload → preview → start button."""
    st.subheader("📋 Batch Onboarding")
    st.markdown(
        "Upload a CSV with columns: **name**, **role** (required), "
        "**github_repo_url**, **drive_folder_url**, **slack_channel_id** (optional)."
    )

    uploaded = st.file_uploader("Upload new-hire CSV", type=["csv"])
    if uploaded is None:
        # Show a sample CSV for convenience
        st.markdown("**Example CSV:**")
        st.code(
            "name,role,github_repo_url,drive_folder_url,slack_channel_id\n"
            "Alice Chen,Backend Engineer,https://github.com/acme/api,,C04TEAM\n"
            "Bob Smith,Frontend Engineer,,,\n",
            language="csv",
        )
        return

    csv_text = uploaded.read().decode("utf-8")
    try:
        hires = parse_csv(csv_text)
    except Exception as exc:
        st.error(f"Failed to parse CSV: {exc}")
        return

    if not hires:
        st.warning("CSV contained no rows.")
        return

    # Preview table
    st.markdown(f"**{len(hires)} new hires parsed:**")
    preview_data = []
    for h in hires:
        preview_data.append({
            "Name": h.name,
            "Role": h.role,
            "GitHub": h.github_repo_url or "—",
            "Drive": h.drive_folder_url or "—",
            "Slack": h.slack_channel_id or "—",
        })
    st.dataframe(preview_data, use_container_width=True)

    if st.button("🚀 Start Batch Onboarding", type="primary"):
        hires_dicts = [asdict(h) for h in hires]
        with st.spinner("Starting Temporal workflow…"):
            try:
                wf_id = _run_async(_start_batch_workflow(hires_dicts))
                st.session_state["batch_workflow_id"] = wf_id
                st.session_state["batch_total"] = len(hires)
                st.success(f"Workflow started: `{wf_id}`")
                st.rerun()
            except Exception as exc:
                st.error(
                    f"Could not start workflow. Is the Temporal server and "
                    f"worker running?\n\n`{exc}`"
                )


def render_batch_progress() -> None:
    """Poll workflow progress and display results."""
    wf_id = st.session_state.get("batch_workflow_id")
    if not wf_id:
        return

    st.subheader("📊 Batch Progress")
    st.caption(f"Workflow: `{wf_id}`")

    try:
        progress = _run_async(_query_progress(wf_id))
        results = _run_async(_query_results(wf_id))
    except Exception as exc:
        st.error(f"Could not query workflow: {exc}")
        return

    total = progress["total"]
    completed = progress["completed"]
    failed = progress["failed"]
    in_progress = progress["in_progress"]

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", total)
    col2.metric("Completed ✅", completed)
    col3.metric("Failed ❌", failed)
    col4.metric("In Progress ⏳", in_progress)

    # Progress bar
    done = completed + failed
    if total > 0:
        st.progress(done / total, text=f"{done}/{total} finished")

    # Per-hire results
    if results:
        st.markdown("---")
        for res in results:
            status_icon = "✅" if res.get("status") == "completed" else "❌"
            with st.expander(
                f"{status_icon} {res.get('name', '?')} — {res.get('role', '?')}"
            ):
                if res.get("status") == "completed":
                    st.markdown(res.get("onboarding_plan", ""))
                else:
                    st.error(f"Error: {res.get('error', 'unknown')}")

    # Auto-refresh while work is still in progress
    if in_progress > 0:
        st.info("Refreshing automatically…")
        import time
        time.sleep(5)
        st.rerun()
    else:
        st.success("🎉 Batch onboarding complete!")
        if st.button("Start New Batch"):
            del st.session_state["batch_workflow_id"]
            del st.session_state["batch_total"]
            st.rerun()
