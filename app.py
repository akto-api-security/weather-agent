import logging
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import AgentError, clear_session, create_weather_agent, invoke_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

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


class ErrorResponse(BaseModel):
    detail: str
    thread_id: str | None = None


@app.exception_handler(AgentError)
async def agent_error_handler(_: Request, exc: AgentError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, responses={502: {"model": ErrorResponse}})
def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        reply = invoke_agent(agent, request.message, thread_id)
    except AgentError:
        raise
    except Exception as exc:
        logger.exception("Unhandled chat error for thread %s", thread_id)
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error",
        ) from exc

    return ChatResponse(reply=reply, thread_id=thread_id)


@app.delete("/sessions/{thread_id}")
def delete_session(thread_id: str):
    clear_session(thread_id)
    return {"status": "deleted", "thread_id": thread_id}
