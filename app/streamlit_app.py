"""Streamlit control panel for the Aviation Content Factory.

Run with::

    streamlit run app/streamlit_app.py

The UI reuses the aviation orchestrator directly (no HTTP layer) and
persists job state to ``data/jobs/`` via ``aviation.persistence``. A
background thread drives the run so Streamlit's script re-runs on
each interaction stay responsive.
"""

from __future__ import annotations

import io
import json
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from aviation.orchestrator import CancelledError, new_job, run_job
from aviation.persistence import delete_job, list_jobs, load_job, save_job
from aviation.state import AviationJob, JobSettings, JobStatus
from core.history import HistoryStore
from core.settings import Settings
from models.aviation_bible import Mode, NarrativeStructure

load_dotenv()

st.set_page_config(
    page_title="Aviation Content Factory",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── shared state ──────────────────────────────────────────────────────


def _settings() -> Settings:
    if "settings" not in st.session_state:
        st.session_state.settings = Settings()
    return st.session_state.settings


def _history() -> HistoryStore:
    if "history" not in st.session_state:
        st.session_state.history = HistoryStore()
    return st.session_state.history


def _cancel_flags() -> dict[str, threading.Event]:
    if "cancel_flags" not in st.session_state:
        st.session_state.cancel_flags = {}
    return st.session_state.cancel_flags


def _threads() -> dict[str, threading.Thread]:
    if "worker_threads" not in st.session_state:
        st.session_state.worker_threads = {}
    return st.session_state.worker_threads


# ── worker ────────────────────────────────────────────────────────────


def _run_worker(job: AviationJob, settings: Settings, history: HistoryStore, cancel_event: threading.Event) -> None:
    """Run a job to completion in a background thread."""
    import asyncio

    def cancel_check() -> bool:
        return cancel_event.is_set()

    try:
        asyncio.run(
            run_job(
                job,
                settings=settings,
                history=history,
                cancel_check=cancel_check,
            )
        )
    except CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        job.append_log("system", str(exc), level="error")
        save_job(job)


def start_job(job: AviationJob) -> None:
    settings = _settings()
    history = _history()
    cancel_event = threading.Event()
    _cancel_flags()[job.job_id] = cancel_event
    save_job(job)
    thread = threading.Thread(
        target=_run_worker,
        args=(job, settings, history, cancel_event),
        daemon=True,
        name=f"aviation-{job.job_id}",
    )
    _threads()[job.job_id] = thread
    thread.start()


def cancel_job(job_id: str) -> None:
    ev = _cancel_flags().get(job_id)
    if ev is not None:
        ev.set()


# ── sidebar ───────────────────────────────────────────────────────────


def _sidebar_settings() -> None:
    st.sidebar.title("Aviation Content Factory")
    st.sidebar.caption("LangGraph-inspired multi-agent pipeline · LiteLLM · Streamlit")

    provider_options = [
        "mock/demo (offline)",
        "openrouter/anthropic/claude-3.5-sonnet",
        "openrouter/openai/gpt-4o",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet-latest",
        "gemini/gemini-1.5-pro",
        "deepseek/deepseek-chat",
    ]
    st.sidebar.markdown("### Default model per role")
    st.sidebar.caption(
        "Leave blank to use the value from settings.yaml. Overrides here "
        "become the default for jobs you create in this session."
    )
    for role in ("primary", "evaluation", "fact_check", "summary", "storyboard"):
        key = f"default_model_{role}"
        current = st.session_state.get(key, "")
        picked = st.sidebar.selectbox(
            role,
            options=[""] + provider_options,
            index=([""] + provider_options).index(current) if current in ([""] + provider_options) else 0,
            key=f"sidebar_{key}",
        )
        st.session_state[key] = "" if picked == "" or picked.startswith("mock/demo") and picked != "mock/demo (offline)" else picked
        if picked == "mock/demo (offline)":
            st.session_state[key] = "mock/demo"

    st.sidebar.divider()
    if st.sidebar.checkbox("Force mock provider (AVIATION_FORCE_MOCK)", key="force_mock"):
        import os as _os
        _os.environ["AVIATION_FORCE_MOCK"] = "1"
    else:
        import os as _os
        _os.environ.pop("AVIATION_FORCE_MOCK", None)

    st.sidebar.divider()
    hist = _history()
    summary = hist.summary()
    st.sidebar.markdown("### Global history")
    st.sidebar.metric("Completed incidents", summary.completed_incidents)
    st.sidebar.write(f"Next narrative structure: **{hist.next_structure().value}**")


# ── pages ─────────────────────────────────────────────────────────────


def page_queue() -> None:
    st.header("Queue a new incident")
    with st.form("new_job", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input(
                "Topic / working title",
                value=st.session_state.get("form_topic", ""),
                placeholder="e.g. Cascading hydraulic failure over the North Atlantic",
            )
            mode = st.radio(
                "Mode",
                options=[Mode.REAL.value, Mode.FICTIONAL.value],
                horizontal=True,
                key="form_mode",
                index=1,
            )
            target_words = st.slider(
                "Target word count", min_value=3000, max_value=20000,
                step=600, value=14400, key="form_target_words",
                help="~150 wpm — 14,400 words ≈ 96 min.",
            )
            chapter_target = st.slider(
                "Chapter target words", min_value=400, max_value=2400,
                step=100, value=1200, key="form_chapter_target",
            )
        with col2:
            min_score = st.slider("Critic min score", 5.0, 10.0, 8.0, 0.5)
            max_revisions = st.slider("Max revisions per chapter", 0, 4, 2)
            max_holistic = st.slider("Max holistic rounds", 0, 3, 2)
            wpm = st.slider("Words per minute (for timestamps)", 100, 200, 150, 10)
            force_structure = st.selectbox(
                "Force structure (override rotation)",
                options=[""] + [s.value for s in NarrativeStructure],
                key="form_force_structure",
            )

        pdfs = st.file_uploader(
            "Source PDFs (real mode only)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        submitted = st.form_submit_button("Add to queue and start", type="primary")

    if submitted:
        if not topic.strip():
            st.error("Topic is required.")
            return
        selected_mode = Mode(mode)
        if selected_mode == Mode.REAL and not pdfs:
            st.error("Real mode needs at least one source PDF or text file.")
            return

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        upload_dir = Path("data/uploads") / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        source_paths: list[str] = []
        for uploaded in pdfs or []:
            path = upload_dir / uploaded.name
            path.write_bytes(uploaded.getbuffer())
            source_paths.append(str(path))

        override: dict = {
            "target_words": int(target_words),
            "chapter_target_words": int(chapter_target),
            "min_score": float(min_score),
            "max_revisions_per_chapter": int(max_revisions),
            "max_holistic_rounds": int(max_holistic),
            "words_per_minute": int(wpm),
        }
        if force_structure:
            override["force_structure"] = NarrativeStructure(force_structure)
        for role in ("primary", "evaluation", "fact_check", "summary", "storyboard"):
            picked = st.session_state.get(f"default_model_{role}", "")
            if picked:
                override[f"model_{role}"] = picked

        job = new_job(
            topic=topic.strip(),
            mode=selected_mode,
            source_pdfs=source_paths,
            settings_override=override,
            job_id=job_id,
        )
        start_job(job)
        st.session_state["last_started_id"] = job_id
        st.success(f"Started {job_id}. Switch to the ‘Live progress’ page to watch.")


def page_progress() -> None:
    st.header("Live progress")
    jobs = list_jobs()
    if not jobs:
        st.info("No jobs yet. Queue one on the ‘New incident’ page.")
        return
    ids = [f"{j.job_id}  ·  {j.title or j.topic[:60]}  ·  {j.status.value}" for j in jobs]
    picked_label = st.selectbox("Job", options=ids, index=0)
    picked_id = picked_label.split()[0]
    try:
        job = load_job(picked_id)
    except FileNotFoundError:
        st.warning("Job disappeared from disk.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", job.status.value)
    col2.metric("Progress", f"{job.progress * 100:.0f} %")
    col3.metric("Cost (USD)", f"{job.cost_usd:.4f}")
    col4.metric("Tokens (in / out)", f"{job.tokens_in:,} / {job.tokens_out:,}")

    st.progress(min(1.0, max(0.0, job.progress)))

    running = job.status == JobStatus.RUNNING
    resumable = job.status in (JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.PAUSED)
    if running:
        if st.button("Cancel", type="secondary"):
            cancel_job(job.job_id)
            st.warning("Cancel signal sent — will stop at the next node boundary.")
    elif resumable:
        if st.button("Resume", type="primary"):
            job.status = JobStatus.RUNNING
            start_job(job)
            st.success("Resumed.")

    st.subheader("Current step")
    st.code(job.current_node or "-", language="")

    st.subheader("Chapters")
    if job.chapters:
        for c in job.chapters:
            label = f"Ch{c.index + 1} · {c.title} · {len(c.draft_text.split()):,}w"
            if c.critic_score:
                label += f" · score {c.critic_score:.1f}"
            if c.approved:
                label += " · ✅"
            elif job.current_chapter == c.index and running:
                label += " · …"
            with st.expander(label, expanded=(job.current_chapter == c.index and running)):
                st.write(", ".join(c.outline_bullets) or "_no outline bullets_")
                if c.critic_notes:
                    st.caption("Critic notes:")
                    st.text(c.critic_notes[:2000])
                if c.fact_check_issues:
                    st.warning("Fact-check issues (HIGH):\n" + "\n".join(f"- {i}" for i in c.fact_check_issues))
                if c.draft_text:
                    st.text(c.draft_text[:4000] + ("…" if len(c.draft_text) > 4000 else ""))
    else:
        st.caption("No chapters yet.")

    st.subheader("Live log")
    logs_reversed = list(reversed(job.logs[-200:]))
    log_text = "\n".join(
        f"{l.ts.split('T')[1] if 'T' in l.ts else l.ts}  [{l.agent:12}] {l.level:7} {l.message}"
        for l in logs_reversed
    )
    st.text_area("logs", value=log_text, height=280, label_visibility="collapsed")

    if job.status == JobStatus.COMPLETED:
        st.subheader("Deliverables")
        for d in job.deliverables:
            p = Path(d.path)
            if p.exists():
                st.download_button(
                    label=f"⬇ {d.filename} ({d.bytes:,} bytes)",
                    data=p.read_bytes(),
                    file_name=d.filename,
                    key=f"dl_{job.job_id}_{d.filename}",
                )
        # Bundle everything as a zip.
        zip_bytes = io.BytesIO()
        with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as zf:
            for d in job.deliverables:
                p = Path(d.path)
                if p.exists():
                    zf.write(p, arcname=d.filename)
        st.download_button(
            "⬇ Download all as .zip",
            data=zip_bytes.getvalue(),
            file_name=f"{job.job_id}.zip",
            mime="application/zip",
            type="primary",
        )

    if running:
        time.sleep(2)
        st.rerun()


def page_history() -> None:
    st.header("Job history")
    jobs = list_jobs()
    if not jobs:
        st.info("Nothing to show yet.")
        return
    for j in jobs:
        with st.expander(
            f"{j.updated_at}  ·  {j.status.value:9}  ·  {j.title or j.topic[:60]}"
        ):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Mode:** {j.settings.mode.value}")
            col1.write(f"**Words target:** {j.settings.target_words:,}")
            col2.write(f"**Chapters:** {len(j.chapters)}")
            col2.write(f"**Tokens:** {j.tokens_in:,} in / {j.tokens_out:,} out")
            col3.write(f"**Cost:** ${j.cost_usd:.4f}")
            col3.write(f"**Structure:** {j.bible.narrative_structure.value if j.bible else '-'}")
            if j.error:
                st.error(j.error)
            if st.button(f"Delete {j.job_id}", key=f"del_{j.job_id}"):
                delete_job(j.job_id)
                out = Path(j.output_dir) if j.output_dir else Path("data/outputs") / j.job_id
                if out.exists():
                    shutil.rmtree(out, ignore_errors=True)
                st.rerun()


def page_global_history() -> None:
    st.header("Global history & rotation")
    hist = _history()
    summary = hist.summary()
    st.write(f"**Completed incidents:** {summary.completed_incidents}")
    st.write(f"**Next structure (forced rotation):** `{hist.next_structure().value}`")
    st.subheader("Structures used")
    if summary.structures_used:
        st.code("\n".join(f"{i + 1}. {s}" for i, s in enumerate(summary.structures_used)))
    else:
        st.caption("None yet.")
    st.subheader("Registered elements (fictional uniqueness)")
    for kind, values in summary.elements_by_kind.items():
        if values:
            with st.expander(f"{kind} ({len(values)})"):
                st.code("\n".join(values))


# ── router ────────────────────────────────────────────────────────────


PAGES = {
    "New incident": page_queue,
    "Live progress": page_progress,
    "History": page_history,
    "Global history": page_global_history,
}


def main() -> None:
    _sidebar_settings()
    page_name = st.sidebar.radio("Page", list(PAGES.keys()))
    PAGES[page_name]()


if __name__ == "__main__":
    main()
