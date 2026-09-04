# ✈ Aviation Content Factory

A LangGraph-inspired, multi-agent Python pipeline that turns a single
incident topic (or an uploaded accident-report PDF) into a **40–60
minute long-form aviation script** for YouTube narration, with three
deliverables ready for the ElevenLabs → Midjourney → post-production
workflow.

Runs entirely offline against a deterministic **mock provider** if no
API keys are configured — the same pipeline runs against real
providers just by setting an environment variable.

## What it produces per incident

| File | Purpose |
|------|---------|
| `01_tts_script.txt` | Clean narration + ElevenLabs `<break>` SSML (0.8 s between paragraphs, 1.5 s after cliffhangers). |
| `02_youtube_metadata.md` | 3–5 title options · hook · ⚠ fictionalization notice · ⏱ chapter timestamps (at 150 wpm) · 📚 sources · 🖋 100–150-word personal note · 20–30 SEO tags. |
| `03_storyboard.csv` (+ pipe & JSON variants) | `Timestamp \| Text Segment \| Image Generation Prompt` — one row per ~5 s. AI-image prompts only, no B-roll suggestions. |
| `00_manuscript.md` | Full source-of-truth manuscript. |
| `04_story_bible.json` | Aviation StoryBible for auditability. |

## Two modes

* **Real Incident (RAG)** — upload NTSB / AAIB / BEA / TSB PDFs. The
  ingest agent walks the report in ~18k-char chunks, extracts
  structured facts (aircraft, timeline, CVR, cause chain, findings),
  and feeds them into every downstream prompt. The Fact-Checker agent
  cross-verifies each chapter's technical claims against the extracted
  facts — a chapter with any HIGH-severity issue is sent back to the
  Editor.
* **Fictional** — invent a realistic incident from the topic. The
  **Global History Manager** guarantees no reused airlines / tail
  numbers / flight numbers / crew names / cities across your batch,
  and rotates the narrative structure (In Media Res → Three-Act →
  Rashomon → Reverse-Chronological → Frame Story) so successive
  stories don't sound the same.

## Architecture

```
                                 [ Batch queue ]
                                        │
                                        ▼
    ingest (real) ─▶ plan ─▶ plan_chapters ─▶ ┌─ write ─────┐
                                              │             │
                             per-chapter loop │  fact-check │  ── HIGH? → editor ─┐
                             (bounded by      │             │                     │
                             max_revisions)   │  critic ────┼── score<min → editor┘
                                              │             │
                                              └─ summarise ─▶ approve → next
                                                                             │
                                              ┌──────────────────────────────┘
                                              ▼
                                       holistic review ── flagged? → editor per ch → holistic (max_holistic_rounds)
                                              │
                                              ▼
                                       post-process → SSML · YouTube MD · Storyboard CSV · manuscript · bible
```

Every node persists the job's full state to
`data/jobs/<job_id>.json` (atomic writes) — a crash or a `Cancel`
click resumes from exactly the last completed node.

The engine sits on:

* **LiteLLM** — one call, every provider. Model IDs use the LiteLLM
  convention (`openrouter/anthropic/claude-3.5-sonnet`, `openai/gpt-4o`,
  `anthropic/claude-3-5-sonnet-latest`, `gemini/gemini-1.5-pro`,
  `deepseek/deepseek-chat`, `kie/<model>`, `custom/<model>`,
  `mock/demo`).
* **Pydantic v2** — every agent's output is a validated model.
* **PyMuPDF** + **pdfplumber** fallback for PDF ingest.
* **SQLite** (single `data/global_history.db`) for cross-story
  uniqueness and structure rotation.
* **Streamlit** for the control panel (queue, live progress,
  history, global-history, settings).

## Quick start

```bash
# 1. Set up the venv (Python 3.11+)
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt

# 2. Configure providers (all optional — mock is the default)
cp .env.example .env
$EDITOR .env

# 3. Launch the UI
.venv/bin/streamlit run app/streamlit_app.py
# → open http://localhost:8501
```

Prefer the CLI? Run a single incident in one command:

```bash
AVIATION_FORCE_MOCK=1 .venv/bin/python -c "
from aviation.orchestrator import new_job, run_job_sync
from models.aviation_bible import Mode
job = new_job(topic='Cascading hydraulic failure over the North Atlantic', mode=Mode.FICTIONAL)
run_job_sync(job)
print('Deliverables in', job.output_dir)
"
```

## Provider setup

Every provider is opt-in. Set the corresponding environment variable
in `.env`:

| Provider | Env var | Model id example |
|----------|---------|------------------|
| OpenAI direct | `OPENAI_API_KEY` | `gpt-4o` or `openai/gpt-4o` |
| Anthropic direct | `ANTHROPIC_API_KEY` | `anthropic/claude-3-5-sonnet-latest` |
| Google Gemini | `GEMINI_API_KEY` | `gemini/gemini-1.5-pro` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/anthropic/claude-3.5-sonnet` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| kie.ai | `KIE_API_KEY` + `KIE_BASE_URL` | `kie/<any-model>` |
| Custom (LiteLLM proxy, vLLM, LM Studio, Ollama…) | `CUSTOM_API_KEY` + `CUSTOM_BASE_URL` | `custom/<any-model>` |
| **Mock** (no key needed) | `AVIATION_FORCE_MOCK=1` | `mock/demo` |

You can point each agent role at a different model in
`settings.yaml` (or override per job in the sidebar): `primary`,
`evaluation`, `fact_check`, `summary`, `storyboard`.

## Batch queue

The Streamlit **New incident** page adds one incident at a time and
starts it in a background thread. You can queue as many as you like —
each runs to completion independently and shares one Global History
Manager, so uniqueness and rotation stay consistent across the batch.

## Repository layout

```
aviation/                # The aviation-specific pipeline
├── state.py             # AviationJob + JobSettings (Pydantic v2)
├── persistence.py       # data/jobs/<id>.json atomic checkpoints
├── orchestrator.py      # run_job() — the whole graph
├── agents.py            # Seven agents, one function each
├── prompts.py           # All prompts (real & fictional)
├── llm_helpers.py       # llm_text / llm_json wrappers
├── deliverables.py      # SSML / YouTube MD / Storyboard CSV
├── text.py              # count_words / segmenter / TTS strip
└── tests/               # 34 unit tests
core/
├── llm/                 # LiteLLM router + offline mock provider
├── history.py           # Global History Manager (SQLite)
├── pdf_ingest.py        # PyMuPDF + pdfplumber
├── api_client.py        # LiteLLM-backed unified client (facade)
└── … (utilities kept from the AI Story Generator base)
models/
├── aviation_bible.py    # AviationStoryBible + ExtractedFacts + …
└── … (generic story models kept from the base project)
app/
└── streamlit_app.py     # Control panel
resources/               # Prompt-template fallbacks (used by legacy runner)
settings.yaml            # Roles → models, retry, rate-limits, pricing
```

## Tests

```bash
.venv/bin/python -m pytest aviation/tests -q            # 34 aviation-specific tests
.venv/bin/python -m pytest tests/unit -q                # foundation tests (603 passing baseline)
```

## Notes

* **Not a real LangGraph graph.** The orchestrator is a plain Python
  state machine that persists a Pydantic checkpoint after every node
  — behaviourally identical to a `StateGraph` + `SqliteSaver`
  combination for this pipeline shape, but easier to reason about.
* **The legacy customtkinter GUI** at `main_legacy_ctk.py` is kept
  for reference; the Streamlit app is the supported UI.
* **The generic story-generator pipeline** (`core/steps/*`,
  `core/step_runner.py`, `resources/prompts/templates/`) is also
  retained — it powers the mock provider's schema-shape hints and
  keeps a broad test suite compiling. You can ignore it if you only
  care about the aviation flow.
