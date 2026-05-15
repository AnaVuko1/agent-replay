# Agent Replay — AI Agent Decision Playback Engine

**Record. Rewind. Replay. Understand why your AI agents make the decisions they do.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-green.svg)](https://fastapi.tiangolo.com/)

## Why Agent Replay?

LLM tracing tools (LangFuse, LangSmith, Helicone) track metrics — latency, tokens, cost. But AI agents aren't LLM calls. They're decision loops: `think → tool call → observe → think → tool call`.

The real problem is **comprehension** — understanding **why** an agent made a decision, not just **that** it made one.

Agent Replay is like session replay for the web, but for AI agents. Record full agent sessions, scrub through the timeline, see the **state** at any moment, and replay decisions with different prompts or models.

## Key Features

- **Session Recording**: Capture agent steps with full context, metadata, and tool interactions
- **Interactive Timeline**: Scrub through agent sessions, jump to any decision point
- **State Snapshots**: See the exact mental state of the agent at any step
- **What-If Analysis**: Replay decisions with different prompts, models, or parameters
- **Comparison Engine**: Side-by-side diff of original vs replayed decisions
- **Dashboard Analytics**: Insights into agent behavior, patterns, and performance
- **Live Streaming**: WebSocket support for real-time agent session monitoring

## Architecture

```
Agent Replay Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Agent      │───▶│   Recorder      │───▶│   Database      │
│   (Any System)  │    │   (API/WS)      │    │   (SQLite)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                        │             │
                                        ▼             ▼
                               ┌─────────────────┐    ┌─────────────────┐
                               │   Timeline      │◀───│   Replay Engine │
                               │   Engine        │    │   (Simulated)   │
                               └─────────────────┘    └─────────────────┘
                                        │                    │
                                        ▼                    ▼
                               ┌─────────────────┐    ┌─────────────────┐
                               │   Web UI        │    │   Comparator    │
                               │   (Dashboard)   │    │   (Diff Engine) │
                               └─────────────────┘    └─────────────────┘
```

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/AnaVuko1/agent-replay.git
cd agent-replay

# Start with Docker Compose
docker-compose up -d

# Visit http://localhost:8000
```

### Local Development

```bash
# 1. Clone and setup
git clone https://github.com/AnaVuko1/agent-replay.git
cd agent-replay

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Seed the database
python seed_data.py

# 6. Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 7. Open http://localhost:8000
```

## API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/sessions` | List all agent sessions |
| POST | `/api/v1/sessions` | Create new session |
| GET | `/api/v1/sessions/{id}` | Get session details |
| POST | `/api/v1/sessions/{id}/steps` | Record a step |
| GET | `/api/v1/sessions/{id}/timeline` | Get timeline with decision cycles |
| GET | `/api/v1/sessions/{id}/snapshot?step={n}` | Get state at specific step |
| POST | `/api/v1/sessions/{id}/replay` | Execute replay with custom config |
| GET | `/api/v1/replays/{id}/compare` | Compare replay vs original |
| WS | `/ws/sessions/{id}/live` | Live step streaming |

## Data Model

### SessionStep Fields
- `step_number`: Sequential step identifier
- `step_type`: `think` | `tool_call` | `tool_result` | `observation` | `decision` | `error`
- `agent_id`: Which agent in multi-agent setup
- `model`: LLM model used (e.g., "claude-3.5-sonnet")
- `content`: Actual thought/tool call/observation
- `metadata`: JSON with tokens_used, latency_ms, tool_name, tool_args
- `context_snapshot`: Agent's full context window (truncated)
- `timestamp`: When the step occurred

### Decision Cycles
Agent Replay groups steps into natural decision cycles:
```
think → tool_call → tool_result → observation → decision
```

Each cycle represents one atomic decision-making process.

## Example Usage

### Recording a Session

```python
import requests
import json

# Create a session
session_response = requests.post(
    "http://localhost:8000/api/v1/sessions",
    json={
        "name": "Debugging Python Error",
        "agent_id": "coding-agent-1",
        "model": "claude-3.5-sonnet"
    }
)
session_id = session_response.json()["id"]

# Record a step
step_response = requests.post(
    f"http://localhost:8000/api/v1/sessions/{session_id}/steps",
    json={
        "step_type": "think",
        "content": "I need to debug this ImportError",
        "metadata": {"tokens_used": 45, "latency_ms": 320}
    }
)
```

### Replaying a Decision

```python
replay_response = requests.post(
    f"http://localhost:8000/api/v1/sessions/{session_id}/replay",
    json={
        "start_step": 15,
        "end_step": 22,
        "replay_config": {
            "model": "gpt-4-turbo",
            "temperature": 0.2,
            "system_prompt": "You are a debugging expert..."
        }
    }
)
replay_id = replay_response.json()["replay_id"]

# Compare with original
compare_response = requests.get(
    f"http://localhost:8000/api/v1/replays/{replay_id}/compare"
)
```

## Dashboard Features

1. **Session Overview**: See all sessions, step counts, decision points
2. **Timeline Scrubber**: Interactive timeline to jump to any step
3. **State Inspector**: View agent context at selected step
4. **Replay Launcher**: Select step range, modify config, run replay
5. **Comparison Viewer**: Side-by-side diff of original vs replayed decisions
6. **Analytics**: Patterns, error rates, decision latency trends

## Integration Guide

### OpenAI / Anthropic Agents
Use the built-in adapters in `app/engine/trace_ingest.py` to integrate with:
- OpenAI Function Calling
- Anthropic Messages API
- LangChain / LlamaIndex agents
- Custom agent frameworks

### WebSocket Live Streaming
Connect to `ws://localhost:8000/ws/sessions/{id}/live` for real-time step streaming.

## Development

### Running Tests

```bash
pytest tests/
```

### Project Structure

```
agent-replay/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration (Pydantic Settings)
│   ├── database.py          # Database connection (aiosqlite)
│   ├── models.py            # SQLAlchemy models
│   ├── schemas.py           # Pydantic schemas
│   ├── routes/              # API endpoints
│   ├── engine/              # Core business logic
│   ├── templates/           # Jinja2 templates
│   └── static/              # CSS/JS assets
├── tests/                   # Test suite
├── seed_data.py            # Demo data generation
└── docker-compose.yml      # Docker configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the need for better AI agent observability
- Built with FastAPI, SQLAlchemy, and modern Python patterns
- Designed for developers building production AI agent systems

---

Built with ❤️ by the AI agent development community.