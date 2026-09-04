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
from aviation.resources import filter_incidents, load_incidents
from aviation.state import AviationJob, JobSettings, JobStatus
from aviation.axes import AXES, SubGenre
from aviation.structures import SPECS, all_structures
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
        # Anthropic (direct)
        "anthropic/claude-3-5-sonnet-latest",
        "anthropic/claude-3-5-haiku-latest",
        # OpenAI (direct)
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        # OpenRouter
        "openrouter/anthropic/claude-3.5-sonnet",
        "openrouter/anthropic/claude-3-haiku",
        "openrouter/openai/gpt-4o",
        # Google Gemini
        "gemini/gemini-1.5-pro",
        "gemini/gemini-1.5-flash",
        # DeepSeek
        "deepseek/deepseek-chat",
        # kie.ai (needs KIE_API_KEY + KIE_BASE_URL in .env)
        "kie/anthropic/claude-3.5-sonnet",
        "kie/anthropic/claude-3-5-haiku-latest",
        "kie/openai/gpt-4o",
        "kie/openai/gpt-4o-mini",
        # Custom OpenAI-compatible endpoint (needs CUSTOM_API_KEY + CUSTOM_BASE_URL)
        "custom/whatever-your-endpoint-serves",
    ]
    st.sidebar.markdown("### Default model per role")
    st.sidebar.caption(
        "Pick a preset or type any LiteLLM model id in the 'custom slug' field "
        "below each role. Leave both blank to fall back to settings.yaml."
    )
    for role in ("primary", "evaluation", "fact_check", "summary", "storyboard"):
        key = f"default_model_{role}"
        current = st.session_state.get(key, "")
        # If a saved value doesn't match a preset (e.g. a hand-typed kie.ai slug),
        # add it to the option list so it stays selected on rerun.
        opts = [""] + provider_options
        if current and current not in opts and current != "mock/demo":
            opts.append(current)
        picked = st.sidebar.selectbox(
            role,
            options=opts,
            index=opts.index(current) if current in opts else 0,
            key=f"sidebar_{key}",
        )
        override = st.sidebar.text_input(
            f"↳ custom slug for {role} (optional)",
            value="",
            placeholder="e.g. kie/anthropic/claude-3-5-sonnet-20241022",
            key=f"sidebar_override_{key}",
        )
        # Priority: custom text > preset > blank. Normalise "mock/demo (offline)".
        if override.strip():
            resolved = override.strip()
        elif picked == "mock/demo (offline)":
            resolved = "mock/demo"
        else:
            resolved = picked
        st.session_state[key] = resolved

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

    # Optional: pick a seed from the catalog before opening the form.
    st.markdown("### Optional: pick a seed incident from the catalog (real mode)")
    seeds = load_incidents()
    seed_labels = ["(none — provide your own PDF or invent a fictional story)"] + [
        f"{inc.name}  ·  {inc.sub_genre_primary}  ·  {inc.monetization_risk}"
        for inc in seeds
    ]
    picked_seed_label = st.selectbox(
        "Seed catalog", options=seed_labels, index=0, key="form_seed_pick",
    )
    picked_seed = None
    if picked_seed_label != seed_labels[0]:
        picked_seed = seeds[seed_labels.index(picked_seed_label) - 1]
        with st.expander("Seed details", expanded=False):
            st.write(f"**Date:** {picked_seed.date}")
            st.write(f"**Aircraft:** {picked_seed.aircraft}")
            st.write(f"**Location:** {picked_seed.location}")
            st.write(f"**Failure:** {picked_seed.failure_type}")
            st.write(f"**Outcome:** {picked_seed.outcome}")
            st.write(f"**Causation:** {picked_seed.causation_type}")
            st.write(f"**Risk:** {picked_seed.monetization_risk}")
            if picked_seed.dramatic_details:
                st.markdown("**Dramatic details:**")
                for d in picked_seed.dramatic_details:
                    st.write(f"- {d}")

    with st.form("new_job", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input(
                "Topic / working title",
                value=st.session_state.get("form_topic", picked_seed.name if picked_seed else ""),
                placeholder="e.g. Cascading hydraulic failure over the North Atlantic",
            )
            mode = st.radio(
                "Mode",
                options=[Mode.REAL.value, Mode.FICTIONAL.value],
                horizontal=True,
                key="form_mode",
                index=0 if picked_seed else 1,
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
            "Source PDFs (real mode only — leave empty to use the seed catalog)",
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

    st.subheader("Axis cooldowns (recent values)")
    st.caption(
        "Values that appear in the recent-history window for their axis are "
        "on cooldown — the Planner will not reuse them until they age out."
    )
    for axis_name, spec in AXES.items():
        recent = summary.axis_recent_values.get(axis_name, [])
        with st.expander(
            f"{axis_name}  (cooldown = {spec.cooldown}, recent = {len(recent)})",
            expanded=False,
        ):
            if recent:
                for i, v in enumerate(recent):
                    st.write(f"{i + 1}. {v}")
            else:
                st.caption("nothing recorded yet")

    st.subheader("Registered fictional elements (uniqueness)")
    for kind, values in summary.elements_by_kind.items():
        if values:
            with st.expander(f"{kind} ({len(values)})"):
                st.code("\n".join(values))


def page_structures() -> None:
    st.header("Narrative structures")
    st.caption(
        "Six documented structures with target quarterly quotas. The Planner "
        "picks one per story; the Global History Manager rotates through them."
    )
    for spec in all_structures():
        with st.expander(
            f"{spec.display_name}  ·  quota {spec.quarterly_quota}/quarter",
            expanded=False,
        ):
            st.markdown(f"**{spec.tagline}**")
            st.write(spec.beat_summary)
            if spec.when_to_use:
                st.markdown("**When to use:**")
                for item in spec.when_to_use:
                    st.write(f"- {item}")
            if spec.when_not_to_use:
                st.markdown("**When NOT to use:**")
                for item in spec.when_not_to_use:
                    st.write(f"- {item}")


def page_seed_catalog() -> None:
    st.header("Seed incident catalog")
    incidents = load_incidents()
    st.caption(f"{len(incidents)} pre-vetted incidents across all sub-genres.")
    # Filter widgets.
    col1, col2 = st.columns(2)
    with col1:
        sub_genre = st.selectbox(
            "Filter by sub-genre",
            options=["(all)"] + sorted({i.sub_genre_primary for i in incidents if i.sub_genre_primary}),
        )
    with col2:
        max_risk = st.selectbox("Max monetization risk", options=["LOW", "MED", "HIGH"], index=2)
    filtered = filter_incidents(
        sub_genre=None if sub_genre == "(all)" else sub_genre,
        max_risk=max_risk,
    )
    st.write(f"**{len(filtered)}** incidents match your filter.")
    for inc in filtered:
        with st.expander(f"{inc.name}  ·  {inc.sub_genre_primary}  ·  {inc.monetization_risk}"):
            st.write(f"**Date:** {inc.date}")
            st.write(f"**Aircraft:** {inc.aircraft}")
            st.write(f"**Location:** {inc.location}")
            st.write(f"**Failure:** {inc.failure_type}")
            st.write(f"**Outcome:** {inc.outcome}")
            st.write(f"**Causation:** {inc.causation_type}")
            st.write(f"**Casualties:** {inc.casualties}")
            if inc.dramatic_details:
                st.markdown("**Dramatic details:**")
                for d in inc.dramatic_details:
                    st.write(f"- {d}")
            if inc.sources:
                st.markdown("**Sources:**")
                for s in inc.sources:
                    st.write(f"- {s}")
            if inc.translation_status == "partial":
                st.info(
                    "Some narrative fields still contain Russian text from the "
                    "original brief. The Planner handles mixed languages fine, "
                    "but you can hand-translate any incident by editing "
                    "`resources/aviation/incidents.yaml`."
                )


# ── router ────────────────────────────────────────────────────────────


PAGES = {
    "New incident": page_queue,
    "Live progress": page_progress,
    "History": page_history,
    "Global history": page_global_history,
    "Seed catalog": page_seed_catalog,
    "Structures": page_structures,
}


def main() -> None:
    _sidebar_settings()
    page_name = st.sidebar.radio("Page", list(PAGES.keys()))
    PAGES[page_name]()


if __name__ == "__main__":
    main()
