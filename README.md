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
| `01_tts_script.txt` | Clean narration + ElevenLabs `<break>` SSML (0.8 s between paragraphs, 1.5 s after cliffhangers) + optional ElevenLabs **v3 audio tags** injected at prosody cues (`[whispers]` / `[urgent]` / `[pause]`). The v3 tags are ignored by v2 renderers, so one file works with either model. |
| `02_youtube_metadata.md` | 3–5 title options · hook · ⚠ fictionalization notice · ⏱ chapter timestamps (at 150 wpm) · 📚 sources · 🖋 100–150-word personal note · 20–30 SEO tags. |
| `03_storyboard.csv` (+ pipe & JSON variants) | `Timestamp \| Text Segment \| Image Generation Prompt` — one row per ~5 s. AI-image prompts only, no B-roll suggestions. |
| `00_manuscript.md` | Full source-of-truth manuscript. |
| `04_story_bible.json` | Aviation StoryBible for auditability. |

## Two modes

* **Real Incident (RAG)** — either upload NTSB / AAIB / BEA / TSB
  PDFs (the ingest agent walks them in ~18k-char chunks and extracts
  structured facts), **or** pick a seed from the built-in catalog of
  **46 pre-vetted real incidents** (US Airways 1549, Gimli Glider,
  BA9, TACA 110, United 232, and 41 more — each with sub-genre,
  causation type, monetization-risk marker, dramatic details, and
  sources). The Fact-Checker agent cross-verifies each chapter's
  technical claims against the extracted facts — a HIGH-severity
  issue sends the chapter back to the Editor.
* **Fictional** — invent a realistic incident from a topic. The
  **Global History Manager** guarantees no reused airlines / tail
  numbers / flight numbers / crew names / cities across your batch,
  applies per-axis **cooldowns** to 8+ rotation axes (sub-genre 2,
  aircraft 5, setting 4, protagonist 6, incident 8, twist 10,
  resolution 6, hook 3-in-10, narrator voice 4, character first name
  15, fictional airline 8, narrative structure 2), enforces the
  **parameter-tuple uniqueness rule** (the triple `{sub_genre,
  aircraft, setting}` must never repeat), and rotates through **6
  narrative structures** (Three-Act / Kishōtenketsu / In Media Res +
  Flashback / Rashomon / Investigation-First / Documentary Braid).
  Fictional stories draw their airline / character names from
  pre-vetted pools (45 airlines by region + 134 character names
  tagged by ethnicity and role hint).

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
├── orchestrator.py      # run_job() — the whole graph, axis picking, seed catalog
├── agents.py            # Seven agents, one function each
├── prompts.py           # All prompts + AVIATION_STYLE_RULES + BLACKLIST_RULES + RETENTION_TIPS
├── axes.py              # 8 rotation axes + enums + quarterly quotas
├── structures.py        # 6 narrative structures with per-structure LLM prompts
├── similarity.py        # Title/opening embedding + noun-overlap checks
├── resources.py         # Seed catalog + name pool + airline pool loaders
├── llm_helpers.py       # llm_text / llm_json wrappers
├── deliverables.py      # SSML (v2 + v3 tags) / YouTube MD / Storyboard CSV
├── text.py              # count_words / segmenter / TTS strip
└── tests/               # 74 unit tests
core/
├── llm/                 # LiteLLM router + offline mock provider
├── history.py           # Global History Manager (SQLite, all 8 axes + seed tracking + tuple uniqueness)
├── pdf_ingest.py        # PyMuPDF + pdfplumber
├── api_client.py        # LiteLLM-backed unified client (facade)
└── … (utilities kept from the AI Story Generator base)
models/
├── aviation_bible.py    # AviationStoryBible + ExtractedFacts + …
└── … (generic story models kept from the base project)
app/
└── streamlit_app.py     # Control panel (Queue · Live · History · Global history · Seed catalog · Structures)
resources/aviation/
├── incidents.yaml       # 46 pre-vetted real aviation incidents
├── character_names.yaml # 134 fictional names tagged by ethnicity + role
└── fictional_airlines.yaml  # 45 fictional airline names by region
scripts/
├── parse_incidents.py    # Re-parse the original .docx → yaml (if the source changes)
└── translate_incidents.py # LLM-driven translation of any incidents left partial (needs a real key)
settings.yaml            # Roles → models, retry, rate-limits, pricing
```

## Tests

```bash
.venv/bin/python -m pytest aviation/tests -q            # 74 aviation-specific tests, all green
```

## Notes

* **Not a real LangGraph graph.** The orchestrator is a plain Python
  state machine that persists a Pydantic checkpoint after every node
  — behaviourally identical to a `StateGraph` + `SqliteSaver`
  combination for this pipeline shape, but easier to reason about.
* **The generic story-generator base** the aviation flow was ported
  from has been removed from the tree; only what aviation actually
  imports remains under `core/` and `models/`. If you ever need the
  generic pipeline back, restore it from the Phase-1 commit history.
