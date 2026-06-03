# Weather Agent (LangChain + Bedrock Mantle)

A simple LangChain agent backed by Amazon Bedrock Mantle via its OpenAI-compatible API. It answers weather questions using a `get_weather` tool (powered by [wttr.in](https://wttr.in/)).

## Prerequisites

- Python 3.10+ (local dev) or Docker
- A Bedrock Mantle API key and base URL for your region

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your Bedrock Mantle credentials in `.env`:

```env
OPENAI_API_KEY=bedrock-api-key-***
OPENAI_BASE_URL=https://bedrock-mantle.ap-south-1.api.aws/v1
OPENAI_MODEL=openai.gpt-oss-20b
PORT=80
```

Default model is `openai.gpt-oss-20b` — the cheapest Bedrock Mantle option (~$0.07/1M input, ~$0.30/1M output vs 2× for 120b).

## Run locally (CLI)

```bash
python agent.py
```

## Run locally (HTTP API)

```bash
uvicorn app:app --reload --port 8000
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Paris?"}'
```

Interactive API docs: http://localhost:8000/docs

## Run with Docker

```bash
cp .env.example .env   # add your Bedrock credentials
docker compose up --build
```

The service listens on `http://localhost:80` (host and container both use port 80).

```bash
curl -X POST http://localhost/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Is it warm in Mumbai right now?"}'
```

Stop the service:

```bash
docker compose down
```

## Project layout

```
agent.py            # Agent factory + CLI
app.py              # FastAPI HTTP server
tools.py            # Weather tool
Dockerfile
docker-compose.yml
requirements.txt
```

## How it works

1. `ChatOpenAI` calls Bedrock Mantle at `/v1/chat/completions`.
2. `create_react_agent` (LangGraph) lets the model decide when to call tools.
3. `get_weather` fetches live data from wttr.in and returns a short summary.
4. LangGraph's in-memory checkpointer stores conversation state per `thread_id`.

## Sessions

Each conversation is keyed by a **`thread_id`**. Reuse the same ID to continue a multi-turn chat.

- First message without `thread_id` → server creates one and returns it
- Follow-up messages → send the same `thread_id` back
- Sessions live in memory inside the container (lost on restart)

Example multi-turn flow:

```bash
# Start a session
curl -X POST http://localhost/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Paris?"}'
# → {"reply": "...", "thread_id": "abc-123"}

# Continue the same session
curl -X POST http://localhost/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What about London?", "thread_id": "abc-123"}'

# End a session (optional)
curl -X DELETE http://localhost/sessions/abc-123
```

When deployed with docker-compose, each container has its own session store. If you scale to multiple replicas later, use a shared checkpointer (e.g. Redis or Postgres).

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/chat` | Send a message, get a reply |
| DELETE | `/sessions/{thread_id}` | Clear a conversation session |

**POST /chat** body:

```json
{
  "message": "What's the weather in Paris?",
  "thread_id": "optional-existing-session-id"
}
```

Response:

```json
{
  "reply": "...",
  "thread_id": "abc-123"
}
```
