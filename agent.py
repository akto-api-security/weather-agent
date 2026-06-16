import logging
import os
import uuid

from akto_middleware import AktoGuardrailsMiddleware
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    class GraphRecursionError(Exception):
        pass

from session_store import (
    SessionStore,
    build_llm_user_message,
    update_session_from_turn,
)
from tools import get_weather

load_dotenv()

logger = logging.getLogger(__name__)

TOOLS = [get_weather]

# Default for ap-south-1 Mantle agent tool-calling (nemotron-nano-9b-v2 often 500/503 here).
DEFAULT_MODEL = "mistral.ministral-3-3b-instruct"
DEFAULT_LLM_TIMEOUT_SECONDS = 90.0

SYSTEM_PROMPT = (
    "You are a helpful weather assistant. "
    "Use the get_weather tool when the user asks about current weather in a city. "
    "If the city is ambiguous, ask a brief clarifying question."
)

session_store = SessionStore()


class AgentError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def _llm_timeout_seconds() -> float:
    raw = os.getenv("LLM_TIMEOUT_SECONDS")
    if raw is None:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid LLM_TIMEOUT_SECONDS=%r, using %s", raw, DEFAULT_LLM_TIMEOUT_SECONDS)
        return DEFAULT_LLM_TIMEOUT_SECONDS
    if value <= 0:
        logger.warning("LLM_TIMEOUT_SECONDS must be positive, using %s", DEFAULT_LLM_TIMEOUT_SECONDS)
        return DEFAULT_LLM_TIMEOUT_SECONDS
    return value


def create_weather_agent():
    """Build a standard LangChain agent with Akto guardrails middleware."""
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    os.environ.setdefault("LANGCHAIN_MODEL", model_name)

    llm = ChatOpenAI(
        model=model_name,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"].strip().rstrip("/"),
        temperature=0,
        timeout=_llm_timeout_seconds(),
        max_retries=0,
    )

    return create_agent(
        llm,
        TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[AktoGuardrailsMiddleware()],
    )


def _normalize_content(content) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "\n".join(part for part in parts if part).strip()
        return text or None
    text = str(content).strip()
    return text or None


def extract_reply(result: dict) -> str:
    messages = result.get("messages") or []
    for message in reversed(messages):
        if type(message).__name__ in {"HumanMessage", "ToolMessage"}:
            continue

        content = _normalize_content(getattr(message, "content", None))
        if content:
            return content

    raise AgentError(502, "Agent returned no text response")


def invoke_agent(agent, message: str, thread_id: str) -> str:
    session = session_store.get_or_create(thread_id)
    llm_message = build_llm_user_message(session, message)

    try:
        result = agent.invoke({"messages": [("user", llm_message)]})
        reply = extract_reply(result)
        update_session_from_turn(session, message, reply, result["messages"])
        return reply
    except AgentError:
        raise
    except ValueError as exc:
        if "Blocked by Akto Guardrails" in str(exc):
            raise AgentError(403, str(exc)) from exc
        raise
    except GraphRecursionError as exc:
        logger.warning("Agent recursion limit hit for thread %s", thread_id)
        raise AgentError(
            502,
            "Agent took too many steps. Try a simpler question.",
        ) from exc
    except Exception as exc:
        logger.exception("Agent invoke failed for thread %s", thread_id)
        raise AgentError(
            502,
            "Weather agent is temporarily unavailable. Please try again.",
        ) from exc


def clear_session(thread_id: str) -> None:
    session_store.clear(thread_id)


def main():
    agent = create_weather_agent()
    thread_id = str(uuid.uuid4())

    print("Weather agent (Bedrock Mantle). Type 'quit' to exit.")
    print(f"Session: {thread_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        try:
            reply = invoke_agent(agent, user_input, thread_id)
        except AgentError as exc:
            print(f"\nAgent error: {exc.message}\n")
            continue

        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    main()
