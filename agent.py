import os
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from session_store import (
    SessionStore,
    build_llm_user_message,
    update_session_from_turn,
)
from tools import get_weather

load_dotenv()

TOOLS = [get_weather]

SYSTEM_PROMPT = (
    "You are a helpful weather assistant. "
    "Use the get_weather tool when the user asks about current weather in a city. "
    "If the city is ambiguous, ask a brief clarifying question."
)

session_store = SessionStore()


def create_weather_agent():
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "openai.gpt-oss-20b"),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=0,
    )

    return create_react_agent(
        llm,
        TOOLS,
        prompt=SYSTEM_PROMPT,
    )


def invoke_agent(agent, message: str, thread_id: str) -> str:
    session = session_store.get_or_create(thread_id)

    llm_message = build_llm_user_message(session, message)

    result = agent.invoke({"messages": [("user", llm_message)]})
    reply = result["messages"][-1].content

    update_session_from_turn(session, message, reply, result["messages"])
    return reply


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

        reply = invoke_agent(agent, user_input, thread_id)
        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    main()
