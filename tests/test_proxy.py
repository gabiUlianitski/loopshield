"""Proxy gateway tests with a mocked upstream."""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from loopshield.proxy import ProxySettings, create_app


def _read_file_turn(call_id: str) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "Read the service configuration.",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "config.txt"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": "FileNotFoundError: config.txt",
        },
    ]


def _app(transport: httpx.BaseTransport, max_tokens: int = 8000) -> TestClient:
    settings = ProxySettings(
        upstream_api_base="https://upstream.test",
        similarity_threshold=0.92,
        max_tokens=max_tokens,
    )
    return TestClient(create_app(settings, transport))


def test_health() -> None:
    client = _app(httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_repeated_tool_loop_is_intercepted_before_upstream() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"id": "should-not-run"})

    client = _app(httpx.MockTransport(handler))
    messages: list[dict[str, object]] = [{"role": "system", "content": "Be careful."}]
    for call_id in ("1", "2", "3"):
        messages.extend(_read_file_turn(call_id))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4.1", "messages": messages},
    )
    body = response.json()

    assert response.status_code == 200
    assert calls == []
    assert body["object"] == "chat.completion"
    assert body["loopshield"]["intercepted"] is True
    assert body["loopshield"]["loop_type"] in {"graph_cycle", "both"}
    nudge = body["choices"][0]["message"]["content"]
    assert nudge == body["loopshield"]["intervention_prompt"]
    assert "`read_file`" in nudge
    assert "different strategy" in nudge


def test_clean_request_is_pruned_and_forwarded() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-up",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                ],
            },
        )

    error = "Error: timeout. " * 80
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Be careful."},
        {"role": "user", "content": "Find the incident."},
        {"role": "assistant", "content": "Searching."},
        {"role": "tool", "name": "search", "content": error},
        {"role": "assistant", "content": "Retrying the search."},
        {"role": "tool", "name": "search", "content": error},
        {"role": "user", "content": "What caused it?"},
        {"role": "assistant", "content": "Waiting to answer."},
    ]
    client = _app(httpx.MockTransport(handler), max_tokens=80)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret"},
        json={"model": "gpt-4.1", "messages": messages},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "done"
    assert len(captured) == 1
    forwarded = json.dumps(captured[0]["messages"])
    assert error not in forwarded
    assert "What caused it?" in forwarded
    assert "Be careful." in forwarded


def test_same_tool_with_different_arguments_is_not_a_loop() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"object": "chat.completion", "choices": []})

    client = _app(httpx.MockTransport(handler))
    messages: list[dict[str, object]] = [{"role": "user", "content": "Audit the configs."}]
    for call_id, path in (("1", "a.txt"), ("2", "b.txt"), ("3", "c.txt")):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": f"contents of {path}"})

    response = client.post("/v1/chat/completions", json={"model": "gpt-4.1", "messages": messages})

    assert response.status_code == 200
    assert "loopshield" not in response.json()
    assert len(calls) == 1


def test_anthropic_tool_cycle_returns_an_anthropic_message() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"type": "message"})

    client = _app(httpx.MockTransport(handler))

    def turn(call_id: str) -> dict[str, object]:
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": "read_file",
                    "input": {"path": "config.txt"},
                }
            ],
        }

    response = client.post(
        "/v1/chat/completions",
        headers={"anthropic-version": "2023-06-01"},
        json={
            "model": "claude-sonnet",
            "max_tokens": 256,
            "system": "Be careful.",
            "messages": [turn("1"), turn("2"), turn("3")],
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert calls == []
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["loopshield"]["intercepted"] is True
    assert "`read_file`" in body["content"][0]["text"]


def test_missing_messages_are_rejected() -> None:
    client = _app(httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    response = client.post("/v1/chat/completions", json={"model": "gpt-4.1"})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
