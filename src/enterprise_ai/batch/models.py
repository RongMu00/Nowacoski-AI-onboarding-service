"""Data models for batch onboarding."""

import csv
import io
from dataclasses import dataclass
from typing import Optional


@dataclass
class NewHire:
    """A single new hire to onboard."""

    name: str
    role: str
    github_repo_url: Optional[str] = None
    drive_folder_url: Optional[str] = None
    slack_channel_id: Optional[str] = None


def parse_csv(csv_content: str) -> list[NewHire]:
    """Parse CSV content into NewHire objects.

    Expected columns: name, role, github_repo_url, drive_folder_url, slack_channel_id
    Only ``name`` and ``role`` are required; the rest default to None.
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    hires: list[NewHire] = []
    for row in reader:
        hires.append(
            NewHire(
                name=row["name"].strip(),
                role=row["role"].strip(),
                github_repo_url=(row.get("github_repo_url") or "").strip() or None,
                drive_folder_url=(row.get("drive_folder_url") or "").strip() or None,
                slack_channel_id=(row.get("slack_channel_id") or "").strip() or None,
            )
        )
    return hires
