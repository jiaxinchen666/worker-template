import runpod
import asyncio
import os
import httpx
from openai import AsyncOpenAI
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

# OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Claude config
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_VERSION = "2023-06-01"

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[dict]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    stop: Optional[List[str]] = None
    user: Optional[str] = None
    stream: Optional[bool] = False
    response_format: Optional[str] = None

class ClaudeChatRequest(BaseModel):
    model: str
    messages: List[dict]
    system: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

async def call_gpt_chat_async(request_body: dict):
    try:
        req = OpenAIChatRequest(**request_body)
        completion = await openai_client.chat.completions.create(**req.dict())
        return {"content": completion.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

async def call_claude_chat_async(request_body: dict):
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": CLAUDE_API_VERSION,
        "content-type": "application/json",
    }

    req = ClaudeChatRequest(**request_body)
    payload = {
        "model": req.model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }

    if req.system:
        payload["system"] = req.system

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(CLAUDE_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return {"content": data["content"][0]["text"]}
    except Exception as e:
        return {"error": str(e)}

async def handler(job):
    """Async handler function that processes jobs."""
    print("start handler")
    job_input = job["input"]["llm_input"]
    llm = job["input"]["llm"]

    if llm == "gpt":
        print("gpt")
        result = await call_gpt_chat_async(job_input)
    elif llm == "claude":
        result = await call_claude_chat_async(job_input)
    else:
        result = {"error": f"Unsupported llm: {llm}"}

    return {"result": result}

runpod.serverless.start({"handler": handler})

