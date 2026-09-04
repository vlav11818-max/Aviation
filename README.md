# AI Story Generator Pro v1.0

Automated evergreen YouTube voiceover story generator with multi-provider
LLM support, step-based pipeline, 4-level quality evaluation, and
parallel processing in 11 languages.

## Features

- **6 API providers** — OpenRouter, OpenAI, Anthropic, Google, DeepSeek, Qwen
- **11 languages** — EN, RU, DE, FR, PT, IT, PL, UK, RO, TR, DA
- **3 pipeline strategies** — auto-selected by target word count:
  - Single shot (< 2 000 words)
  - Two pass (2 000–4 000 words)
  - Full pipeline (> 4 000 words) with section-by-section generation
- **4-level evaluation** — Technical, Linguistic, Content, Voiceover
- **Evaluate → revise loop** — automatic revision until quality threshold (default 9.0/10)
- **5 story structures** — Three Act, Hero's Journey, In Medias Res, Episodic, Circular
- **Text adaptation** — translate/adapt existing stories in 3 modes (literal, cultural, free)
- **Parallel processing** — N concurrent workers with per-provider rate limiting
- **Crash recovery** — auto-save after every step, resume from interruption
- **Result caching** — skip already-processed topics
- **Export** — TXT (clean text) and SSML (for TTS systems)
- **Analytics** — per-language, per-provider, score distribution, quality trends
- **GUI** — customtkinter dark-theme interface with progress tracking, log, and analytics

## Quick Start

### 1. Clone and install

```bash
git clone <repository-url>
cd ai-story-generator-pro
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the example environment file and add your keys:

```bash
cp .env.example .env
```

Edit `.env` and fill in at least one provider key:

```
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
DEEPSEEK_API_KEY=sk-...
QWEN_API_KEY=sk-...
```

### 3. Run

```bash
python main.py
```

The GUI will open. Select a topics file, choose your language and style,
configure the API provider, and click **START**.

## Configuration

All settings are in `settings.yaml`. The file is loaded at startup and
merged with defaults from `resources/defaults/settings.yaml`.

Key sections:

| Section | Description |
|---------|-------------|
| `api` | Default provider, model, timeout, retry settings, pricing table |
| `generation` | Default tone, perspective, register, pacing, target words, min score, max attempts |
| `strategy` | Word-count thresholds for strategy auto-selection |
| `parallelism` | Max threads, auto-throttle |
| `paths` | Output, data, resources, recovery, cache directories |
| `cache` | Enable/disable caching, skip-processed flag |
| `ssml` | Pause durations (sentence, paragraph, section) |
| `logging` | Log level, file rotation settings |

## Supported Providers

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| OpenRouter | `OPENROUTER_API_KEY` | anthropic/claude-3.5-sonnet, meta-llama/llama-3.1-70b, google/gemini-pro-1.5, mistralai/mistral-large |
| OpenAI | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022 |
| Google | `GOOGLE_API_KEY` | gemini-1.5-pro, gemini-1.5-flash |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat, deepseek-coder |
| Qwen | `QWEN_API_KEY` | qwen-max, qwen-plus, qwen-turbo |

## Supported Languages

| Code | Language | Flag |
|------|----------|------|
| en | English | 🇬🇧 |
| ru | Russian | 🇷🇺 |
| de | German | 🇩🇪 |
| fr | French | 🇫🇷 |
| pt | Portuguese | 🇵🇹 |
| it | Italian | 🇮🇹 |
| pl | Polish | 🇵🇱 |
| uk | Ukrainian | 🇺🇦 |
| ro | Romanian | 🇷🇴 |
| tr | Turkish | 🇹🇷 |
| da | Danish | 🇩🇰 |

Each language has dedicated cultural instruction files and prompt
localisation to ensure natural, culturally appropriate output.

## Usage

### Generation Mode

1. Create a text file with one topic per line (e.g., `topics.txt`)
2. Open the application and select **Generate** mode
3. Click **Browse** and select your topics file
4. Choose the output language
5. Configure style (tone, perspective, register, length, structure)
6. Select API provider and model
7. Click **START**

The application will show a cost estimate before proceeding.
Stories are saved to the `output/` directory, organised by language
and topic.

### Adaptation Mode

1. Place source `.txt` files in a folder
2. Select **Adapt** mode in the application
3. Choose the source folder
4. Select target language(s)
5. Choose adaptation mode:
   - **Literal** — faithful translation preserving structure
   - **Cultural** — localised adaptation with cultural references
   - **Free** — creative reimagining for the target culture
6. Configure adaptation parameters and click **START**

## Output Structure

Each completed story produces:

```
output/<language>/<topic>/
├── concept.json          # Story concept and bible
├── outline.json          # Structural outline
├── section_1.txt ...     # Individual sections (full pipeline only)
├── draft_v1.txt          # First draft
├── draft_v2.txt          # Revised draft(s)
├── eval_v1.json          # Evaluation results
├── final.txt             # Final clean text
├── final.ssml            # SSML markup for TTS
└── metadata.json         # Full run metadata
```

## Project Structure

```
ai-story-generator-pro/
├── main.py                    # Entry point
├── settings.yaml              # User configuration
├── .env                       # API keys (not committed)
├── core/                      # Core logic
│   ├── steps/                 # Pipeline step implementations
│   ├── api_adapters/          # Provider-specific API adapters
│   ├── api_client.py          # Unified API client
│   ├── step_runner.py         # Strategy executor
│   ├── state_manager.py       # Pipeline state management
│   ├── prompt_manager.py      # Template loading and rendering
│   ├── parallel_processor.py  # Concurrent batch processing
│   ├── analytics_collector.py # Analytics persistence
│   └── ...                    # Events, settings, cache, recovery
├── models/                    # Pydantic data models
├── gui/                       # customtkinter GUI panels
├── exporters/                 # TXT, SSML, report exporters
├── resources/                 # Prompts, cultural files, structures
├── utils/                     # Logging, file I/O, token counting
├── tests/                     # Unit and integration tests
└── data/                      # Analytics, cache, recovery state
```

## Development

### Install development dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/ -v
```

### Run tests with coverage

```bash
pytest tests/ --cov=core --cov=models --cov=utils --cov=exporters -v
```

### Code style

- Python 3.11+
- Type hints on all functions (parameters and return types)
- Google-style docstrings on all classes and public methods
- Absolute imports from project root
- `logging.getLogger(__name__)` in every module (no `print()`)
- Custom exceptions from `core/exceptions.py`
- Pydantic v2 for all data models

## License

*License placeholder — to be determined.*
