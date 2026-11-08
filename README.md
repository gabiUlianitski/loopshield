# LoopShield

LoopShield is a vendor-agnostic Python SDK and an OpenAI-compatible proxy that stops AI agents from burning tokens on the same tool cycle or the same thought. The proxy sits in front of the model provider. Every `POST /v1/chat/completions` is screened before it is forwarded. A detected loop comes back as a normal assistant message that tells the agent the repeating tool is locked and it must change strategy. The upstream provider is not called.

The core package depends only on `networkx`, `numpy`, and Pydantic v2. FastAPI, uvicorn, and httpx are an optional `proxy` extra. Embedding models are injected; the OpenAI and Ollama SDKs are not core dependencies.

## How a call is intercepted

Clients keep speaking the provider’s chat API. LoopShield accepts an OpenAI Chat Completions body, and it also accepts an Anthropic Messages body on the same route. Anthropic shape is recognized from an `anthropic-version` header, a payload that has both `system` and `max_tokens`, or content blocks of type `tool_use` / `tool_result`.

```mermaid
flowchart TD
    agent["Agent or app"]
    route["POST /v1/chat/completions"]
    style{"Payload style"}
    steps["Assistant turns become AgentSteps"]
    inspect["LoopShield.inspect"]
    graph["GraphAnalyzer<br/>exact tool-call cycle"]
    semantic["SemanticAnalyzer<br/>cosine similarity of thoughts"]
    loop{"Loop?"}
    nudge["Intercepted assistant message<br/>intervention_prompt"]
    budget{"Over SHIELD_MAX_TOKENS?"}
    prune["prune_context<br/>keep system prompt and last 2 turns"]
    openai["Upstream POST /v1/chat/completions"]
    anthropic["Upstream POST /v1/messages"]

    agent --> route --> style --> steps --> inspect
    inspect --> graph
    inspect --> semantic
    graph --> loop
    semantic --> loop
    loop -->|yes| nudge
    nudge --> agent
    loop -->|no| budget
    budget -->|yes| prune --> forward{"Provider"}
    budget -->|no| forward
    forward -->|OpenAI| openai
    forward -->|Anthropic| anthropic
    openai --> agent
    anthropic --> agent
```

A clean request is forwarded with the caller’s `Authorization`, `x-api-key`, `anthropic-version`, `anthropic-beta`, `openai-organization`, and `openai-project` headers. LoopShield does not store those credentials.

When a loop is found, the JSON response still looks like a completion:

- OpenAI: `object` is `chat.completion`, and `choices[0].message.content` is the nudge.
- Anthropic: `type` is `message`, `role` is `assistant`, and the text block is the nudge.
- Both include `loopshield.intercepted`, `loopshield.loop_type` (`graph_cycle`, `semantic`, or `both`), and `loopshield.intervention_prompt`.

`stream: true` on an intercepted call is returned as server-sent events with the same nudge. `GET /health` returns `{"status": "ok"}`.

## Quick start

Build and run the gateway on port 8000. Point `UPSTREAM_API_BASE` at the provider you want clean calls to reach.

```bash
docker build -t loopshield:0.1.0 .
docker run --rm -p 8000:8000 -e UPSTREAM_API_BASE=https://api.openai.com -e SHIELD_SIMILARITY_THRESHOLD=0.92 loopshield:0.1.0
```

Check the process, then send a chat completion through it. The provider key stays on the request; the container only forwards it.

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"gpt-4.1\",\"messages\":[{\"role\":\"user\",\"content\":\"Summarize the deploy log.\"}]}"
```

An OpenAI SDK client uses the proxy as its base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-...")
completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Summarize the deploy log."}],
)
```

Anthropic-shaped traffic uses the same path. Set the upstream base to `https://api.anthropic.com` and send the provider headers the proxy already forwards:

```bash
docker run --rm -p 8000:8000 -e UPSTREAM_API_BASE=https://api.anthropic.com loopshield:0.1.0
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"claude-sonnet-4-5\",\"max_tokens\":256,\"system\":\"Be careful.\",\"messages\":[{\"role\":\"user\",\"content\":\"Summarize the deploy log.\"}]}"
```

The official Anthropic SDK posts to `/v1/messages`. This gateway accepts that payload shape on `/v1/chat/completions`, then forwards a clean call to `{UPSTREAM_API_BASE}/v1/messages`. If the upstream base already ends in `/v1`, that prefix is not doubled.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `UPSTREAM_API_BASE` | `https://api.openai.com` | Origin for forwarded calls. OpenAI traffic goes to `/v1/chat/completions`. Anthropic traffic goes to `/v1/messages`. |
| `SHIELD_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity, from 0 to 1, above which two thoughts count as the same line of reasoning. The comparison is strict `>`. |
| `SHIELD_MAX_TOKENS` | `8000` | Estimated token budget. Over this, `prune_context` keeps the leading system prompt and the last two turns, and collapses repeated tool errors in the middle. |
| `SHIELD_MAX_CYCLES` | `1` | A directed tool cycle is a loop when it completes more times than this. `1` allows one lap and flags the second. `0` flags the first time the cycle closes. |
| `SHIELD_MAX_SEMANTIC_REPEATS` | `2` | Minimum streak of consecutive similar thoughts, measured backward from the latest thought, required to flag a semantic loop. |
| `SHIELD_UPSTREAM_TIMEOUT` | `60` | Upstream HTTP timeout in seconds. |

The image sets `UPSTREAM_API_BASE`, `SHIELD_SIMILARITY_THRESHOLD`, and `SHIELD_MAX_TOKENS`. Override any of them with `docker run -e`.

Two detectors run on every request:

- **Graph.** Each tool call is a node of `tool_name` plus normalized arguments. Consecutive calls are directed edges. The same cycle walked again, including a tool calling itself with the same arguments, is a graph loop.
- **Semantic.** Thoughts are compared with cosine similarity. With no embedding model configured, the proxy uses an in-process TF-IDF bag-of-words embedder. Identical text scores 1.0. A streak breaks at the first thought that does not exceed the threshold.

The proxy calls `inspect`, which does not keep request history on the shared process. Concurrent requests do not contaminate each other.

## Use the SDK in process

Install the library when the agent loop lives in your own code and you do not need the gateway:

```bash
pip install -e ".[dev]"
```

Record each step. A verdict with `is_loop` carries the system nudge to send back to the model.

```python
from loopshield import AgentStep, LoopShield

shield = LoopShield(similarity_threshold=0.92, max_cycles=1)

verdict = shield.step(
    AgentStep(
        thought="The config file should be at the project root.",
        tool_name="read_file",
        tool_args={"path": "config.txt"},
        tokens_used=12,
    )
)

if verdict.is_loop:
    print(verdict.intervention_prompt)
```

`@shield.guard` wraps a function that takes an `AgentStep`. When the new step closes a loop, the decorator returns the `LoopVerdict` and does not call the function. Pass an initialized OpenAI or Ollama client as `embedding=` to replace the TF-IDF fallback; pass `embed_fn=` for any other model. Do not pass both.

`prune_context(messages, max_tokens_estimate)` takes OpenAI-style message dicts. It estimates tokens from `token_count`, `tokens`, or `usage.total_tokens` when present, otherwise from character length at about four characters per token.

## Framework integrations

`loopshield/frameworks.py` plugs the same engine into LangChain, CrewAI, and LangGraph. Each framework is an optional install (`pip install "loopshield[langchain]"`, `[langgraph]`, `[crewai]`). The hooks keep no history on the shield, so one `LoopShield` can serve concurrent runs.

**LangChain.** `shield.langchain_middleware(config)` returns an `AgentMiddleware`:

```python
from langchain.agents import create_agent

agent = create_agent(model, tools, middleware=[shield.langchain_middleware({"max_tokens": 6000})])
```

- `wrap_tool_call` replaces a looping tool call with an error `ToolMessage` that carries the nudge. The tool does not run, and the model reads why on its next turn.
- `wrap_model_call` runs `prune_context` when the transcript is over budget. For a loop that reaches the model, it answers with the nudge before the LLM is called. Set `on_loop` to `inject` to append the nudge as a system message and let the model run.

**CrewAI.** `@shield.crewai_agent()` wraps `Agent.execute_task`, either on a class or on the method. It rebuilds the tool trajectory from `agent.tools_results`, or from the `Action` / `Action Input` lines in the string context. When the tools loop, the nudge is prepended to the `context` string that the next step reads.

```python
@shield.crewai_agent()
class GuardedAgent(Agent): ...
```

**LangGraph.** `shield.langgraph_node()` wraps a node. A looping `state["messages"]` gets the system nudge as a state update, and the node is skipped for that pass. With `{"retry_node": "agent"}` it returns `Command(update=..., goto="agent")`, so the graph re-enters that node with the nudge in context. The `messages` channel needs an appending reducer such as `add_messages`.

```python
graph.add_node("agent", shield.langgraph_node(agent_node, config={"retry_node": "agent"}))
```

## Development

```bash
python -m pytest
```

Python 3.11 or newer. Tests live in `tests/` and cover tool-call cycles, semantic streaks, recovery after a nudge, context pruning, embedding adapters, and the proxy with a mocked upstream.

| Module | Role |
| --- | --- |
| `loopshield/models.py` | `AgentStep` and `LoopVerdict` |
| `loopshield/graph_analyzer.py` | Directed tool-call cycles |
| `loopshield/semantic_analyzer.py` | Cosine similarity of thoughts |
| `loopshield/mitigator.py` | Intervention nudge and context pruning |
| `loopshield/shield.py` | `LoopShield.step`, `inspect`, and `guard` |
| `loopshield/adapters.py` | Optional OpenAI and Ollama embedding wrappers |
| `loopshield/proxy.py` | FastAPI gateway served by uvicorn |
| `loopshield/transcript.py` | Chat messages from any framework to `AgentStep`s |
| `loopshield/frameworks.py` | LangChain, CrewAI, and LangGraph integrations |
