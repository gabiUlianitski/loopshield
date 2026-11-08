"""OpenAI- and Anthropic-compatible proxy that screens agent loops before upstream calls."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from loopshield.mitigator import estimate_tokens, prune_context
from loopshield.shield import LoopShield
from loopshield.transcript import content_text as _content_text
from loopshield.transcript import messages_to_steps as _messages_to_steps
from loopshield.transcript import nudge_for as _nudge_for

_FORWARD_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "openai-organization",
    "openai-project",
)


class ProxySettings(BaseModel):
    """Runtime configuration loaded from the process environment."""

    model_config = ConfigDict(frozen=True)

    upstream_api_base: str = "https://api.openai.com"
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    max_tokens: int = Field(default=8000, ge=1)
    max_cycles: int = Field(default=1, ge=0)
    max_semantic_repeats: int = Field(default=2, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)

    @classmethod
    def from_env(cls) -> ProxySettings:
        """Read the proxy settings from environment variables."""

        return cls(
            upstream_api_base=os.environ.get("UPSTREAM_API_BASE", "https://api.openai.com"),
            similarity_threshold=float(os.environ.get("SHIELD_SIMILARITY_THRESHOLD", "0.92")),
            max_tokens=int(os.environ.get("SHIELD_MAX_TOKENS", "8000")),
            max_cycles=int(os.environ.get("SHIELD_MAX_CYCLES", "1")),
            max_semantic_repeats=int(os.environ.get("SHIELD_MAX_SEMANTIC_REPEATS", "2")),
            timeout_seconds=float(os.environ.get("SHIELD_UPSTREAM_TIMEOUT", "60")),
        )


def create_app(
    settings: ProxySettings | None = None,
    transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    """Build the proxy application.

    ``transport`` is for tests that need to capture the upstream request.
    Uvicorn serves the module-level ``app`` created from the environment.
    """

    resolved = settings or ProxySettings.from_env()
    application = FastAPI(title="LoopShield Proxy", version="0.1.0")
    application.state.settings = resolved
    application.state.transport = transport
    application.state.shield = LoopShield(
        similarity_threshold=resolved.similarity_threshold,
        max_cycles=resolved.max_cycles,
        max_semantic_repeats=resolved.max_semantic_repeats,
        max_tokens_estimate=resolved.max_tokens,
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        return await _handle_completion(request, application)

    return application


app = create_app()


async def _handle_completion(request: Request, application: FastAPI) -> JSONResponse | StreamingResponse:
    settings: ProxySettings = application.state.settings
    payload = await _read_payload(request)
    if isinstance(payload, JSONResponse):
        return payload

    style = _api_style(request, payload)
    messages = _normalize_messages(payload, style)
    if isinstance(messages, JSONResponse):
        return messages

    steps = _messages_to_steps(messages)
    verdict = application.state.shield.inspect(steps)
    if verdict.is_loop:
        nudge = _nudge_for(verdict, steps)
        if bool(payload.get("stream")):
            return StreamingResponse(
                _intercepted_stream(style, str(payload.get("model") or "loopshield"), nudge, verdict.loop_type),
                media_type="text/event-stream",
            )
        body = _intercepted_body(style, str(payload.get("model") or "loopshield"), nudge, verdict.loop_type, messages)
        return JSONResponse(body)

    screened = _screen_messages(messages, settings.max_tokens)
    outbound = _outbound_payload(payload, screened, style)
    return await _forward(request, settings, application.state.transport, outbound, style)


async def _read_payload(request: Request) -> dict[str, Any] | JSONResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return _error("Request body must be valid JSON.", anthropic=False, status_code=400)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", anthropic=False, status_code=400)
    return payload


def _api_style(request: Request, payload: Mapping[str, Any]) -> str:
    if request.headers.get("anthropic-version"):
        return "anthropic"
    if "system" in payload and "max_tokens" in payload:
        return "anthropic"
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, list) and any(
            isinstance(part, Mapping) and part.get("type") in {"tool_use", "tool_result"}
            for part in content
        ):
            return "anthropic"
    return "openai"


def _normalize_messages(
    payload: Mapping[str, Any],
    style: str,
) -> list[dict[str, Any]] | JSONResponse:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return _error("messages must be a non-empty list.", anthropic=style == "anthropic", status_code=400)
    if any(not isinstance(message, dict) for message in raw_messages):
        return _error("every message must be an object.", anthropic=style == "anthropic", status_code=400)

    messages = [dict(message) for message in raw_messages]
    if style == "anthropic":
        system = payload.get("system")
        if isinstance(system, str) and system.strip():
            messages.insert(0, {"role": "system", "content": system})
        elif isinstance(system, list) and system:
            messages.insert(0, {"role": "system", "content": system})
    if not str(payload.get("model") or "").strip():
        return _error("model is required.", anthropic=style == "anthropic", status_code=400)
    return messages


def _screen_messages(messages: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    if estimate_tokens(messages) <= max_tokens:
        return messages
    return prune_context(messages, max_tokens)


def _outbound_payload(
    payload: Mapping[str, Any],
    messages: list[dict[str, Any]],
    style: str,
) -> dict[str, Any]:
    outbound = dict(payload)
    if style == "openai":
        outbound["messages"] = messages
        return outbound
    system, rest = _split_leading_system(messages)
    if system:
        contents = [_content_text(message.get("content")) for message in system]
        outbound["system"] = "\n\n".join(part for part in contents if part)
    else:
        outbound.pop("system", None)
    outbound["messages"] = [_anthropic_message(message) for message in rest]
    return outbound


def _anthropic_message(message: dict[str, Any]) -> dict[str, Any]:
    if str(message.get("role", "")).casefold() != "system":
        return message
    summary = _content_text(message.get("content"))
    return {"role": "user", "content": f"LoopShield context summary: {summary}"}


def _split_leading_system(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = 0
    while index < len(messages) and str(messages[index].get("role", "")).casefold() == "system":
        index += 1
    return messages[:index], messages[index:]


async def _forward(
    request: Request,
    settings: ProxySettings,
    transport: httpx.BaseTransport | None,
    payload: dict[str, Any],
    style: str,
) -> JSONResponse | StreamingResponse:
    path = "/v1/messages" if style == "anthropic" else "/v1/chat/completions"
    url = _upstream_url(settings.upstream_api_base, path)
    headers = _forward_headers(request)
    if payload.get("stream") is True:
        return StreamingResponse(
            _stream_upstream(url, payload, headers, settings, transport),
            media_type="text/event-stream",
        )
    try:
        async with httpx.AsyncClient(transport=transport, timeout=settings.timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        return _error(f"Upstream request failed: {exc}", anthropic=style == "anthropic", status_code=502)
    return _upstream_json(response, style)


def _upstream_json(response: httpx.Response, style: str) -> JSONResponse:
    try:
        body = response.json()
    except json.JSONDecodeError:
        body = {"error": {"message": response.text, "type": "upstream_error"}}
    return JSONResponse(body, status_code=response.status_code)


async def _stream_upstream(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    settings: ProxySettings,
    transport: httpx.BaseTransport | None,
) -> AsyncIterator[bytes]:
    client = httpx.AsyncClient(transport=transport, timeout=settings.timeout_seconds)
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            async for chunk in response.aiter_raw():
                yield chunk
    except httpx.HTTPError as exc:
        message = json.dumps(_error_body(f"Upstream request failed: {exc}", anthropic=False))
        yield f"data: {message}\n\n".encode()
    finally:
        await client.aclose()


def _forward_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name in _FORWARD_HEADERS:
        value = request.headers.get(name)
        if value:
            headers[name] = value
    return headers


def _upstream_url(base: str, path: str) -> str:
    root = base.strip().rstrip("/")
    if not root:
        raise ValueError("UPSTREAM_API_BASE is required")
    if root.endswith("/v1") and path.startswith("/v1/"):
        return f"{root}{path[len('/v1'):]}"
    return f"{root}{path}"


def _intercepted_body(
    style: str,
    model: str,
    nudge: str,
    loop_type: str | None,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_tokens = estimate_tokens(messages)
    completion_tokens = max(1, (len(nudge) + 3) // 4)
    marker = {
        "intercepted": True,
        "loop_type": loop_type,
        "intervention_prompt": nudge,
    }
    if style == "anthropic":
        return {
            "id": f"msg_loopshield_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": nudge}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
            "loopshield": marker,
        }
    return {
        "id": f"chatcmpl-loopshield-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": nudge},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "loopshield": marker,
    }


async def _intercepted_stream(
    style: str,
    model: str,
    nudge: str,
    loop_type: str | None,
) -> AsyncIterator[bytes]:
    if style == "anthropic":
        events = [
            {"type": "message_start", "message": {"id": "msg_loopshield", "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "usage": {"input_tokens": 0, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": nudge}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": max(1, (len(nudge) + 3) // 4)}},
            {"type": "message_stop"},
        ]
        for event in events:
            name = str(event["type"])
            yield f"event: {name}\ndata: {json.dumps(event)}\n\n".encode()
        return

    created = int(time.time())
    completion_id = f"chatcmpl-loopshield-{uuid.uuid4().hex[:24]}"
    chunks = [
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": nudge}, "finish_reason": None}],
            "loopshield": {"intercepted": True, "loop_type": loop_type, "intervention_prompt": nudge},
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _error(message: str, *, anthropic: bool, status_code: int) -> JSONResponse:
    return JSONResponse(_error_body(message, anthropic=anthropic), status_code=status_code)


def _error_body(message: str, *, anthropic: bool) -> dict[str, Any]:
    if anthropic:
        return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return {"error": {"message": message, "type": "invalid_request_error", "code": "invalid_request_error"}}
