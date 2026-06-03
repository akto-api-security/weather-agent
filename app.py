import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import clear_session, create_weather_agent, invoke_agent

load_dotenv()

agent = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agent
    agent = create_weather_agent()
    yield


app = FastAPI(title="Weather Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = Field(
        default=None,
        description="Conversation session ID. Omit to start a new session.",
    )


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    thread_id = request.thread_id or str(uuid.uuid4())
    result = invoke_agent(agent, request.message, thread_id)
    reply = result["messages"][-1].content
    return ChatResponse(reply=reply, thread_id=thread_id)


@app.delete("/sessions/{thread_id}")
def delete_session(thread_id: str):
    clear_session(thread_id)
    return {"status": "deleted", "thread_id": thread_id}
