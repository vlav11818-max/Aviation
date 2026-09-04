"""Job persistence — atomic JSON writes for crash-recovery.

Each :class:`~aviation.state.AviationJob` is written to
``data/jobs/<job_id>.json`` after every node boundary. Reading a job
back reconstitutes the full Pydantic tree. This is the aviation
factory's equivalent of a LangGraph SqliteSaver checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable

from aviation.state import AviationJob

logger = logging.getLogger(__name__)

DEFAULT_JOBS_DIR = Path("data/jobs")


def jobs_dir(base: Path | str | None = None) -> Path:
    d = Path(base) if base else DEFAULT_JOBS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_job(job: AviationJob, base: Path | str | None = None) -> Path:
    """Atomically write a job to disk. Returns the file path."""
    directory = jobs_dir(base)
    path = directory / f"{job.job_id}.json"
    data = job.model_dump(mode="json")
    # Write to a temp file in the same directory then os.replace — atomic
    # on POSIX and NTFS.
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup of the tmp file on failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_job(job_id: str, base: Path | str | None = None) -> AviationJob:
    """Load a job by id; raises FileNotFoundError if missing."""
    path = jobs_dir(base) / f"{job_id}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return AviationJob.model_validate(data)


def list_jobs(base: Path | str | None = None) -> list[AviationJob]:
    """Return every job on disk, newest updated first."""
    out: list[AviationJob] = []
    for p in jobs_dir(base).glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            out.append(AviationJob.model_validate(data))
        except Exception as exc:
            logger.warning("Skipping unreadable job file %s: %s", p, exc)
    out.sort(key=lambda j: j.updated_at, reverse=True)
    return out


def delete_job(job_id: str, base: Path | str | None = None) -> bool:
    path = jobs_dir(base) / f"{job_id}.json"
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def touching_save(jobs: Iterable[AviationJob], base: Path | str | None = None) -> None:
    """Bulk-save a set of jobs (used by the queue driver)."""
    for job in jobs:
        save_job(job, base)
