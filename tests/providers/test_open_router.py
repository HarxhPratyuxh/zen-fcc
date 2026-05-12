"""Tests for OpenRouter provider (OpenAI-compatible chat completions)."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
    thinking_content,
)
from providers.base import ProviderConfig
from providers.open_router import OpenRouterProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "minimax-m2.5-free"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.tool_choice = None
        self.metadata = None
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def open_router_config():
    return ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://opencode.ai/zen/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def open_router_provider(open_router_config):
    with patch("providers.openai_compat.AsyncOpenAI"):
        return OpenRouterProvider(open_router_config)


def test_init(open_router_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = OpenRouterProvider(open_router_config)
        assert provider._api_key == "test_openrouter_key"
        assert provider._base_url == "https://opencode.ai/zen/v1"
        mock_openai.assert_called_once()


def test_init_uses_configurable_timeouts():
    """Provider passes configurable read/write/connect timeouts to AsyncOpenAI."""
    config = ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://opencode.ai/zen/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        OpenRouterProvider(config)
        timeout = mock_openai.call_args.kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_build_request_body_is_openai_format(open_router_provider):
    req = MockRequest()
    body = open_router_provider._build_request_body(req)

    assert body["model"] == "minimax-m2.5-free"
    assert body["temperature"] == 0.5
    assert isinstance(body["messages"], list)
    # System prompt becomes a system message + the user message
    assert len(body["messages"]) >= 2
    assert "extra_body" not in body


def test_build_request_body_omits_reasoning_when_globally_disabled(
    open_router_config,
):
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = OpenRouterProvider(
            open_router_config.model_copy(update={"enable_thinking": False})
        )

    body = provider._build_request_body(MockRequest())

    # When thinking is disabled, no reasoning fields should appear
    assert "reasoning" not in body


def test_build_request_body_omits_reasoning_when_request_disables_thinking(
    open_router_provider,
):
    req = MockRequest()
    req.thinking.enabled = False

    body = open_router_provider._build_request_body(req)

    assert "reasoning" not in body


def test_build_request_body_omits_reasoning_when_native_thinking_disabled(
    open_router_provider,
):
    req = MockRequest(thinking={"type": "disabled"})

    body = open_router_provider._build_request_body(req)

    assert "reasoning" not in body


def _make_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
    usage: object | None = None,
):
    """Build a minimal OpenAI streaming chunk SimpleNamespace."""
    delta_kwargs: dict = {}
    if content is not None:
        delta_kwargs["content"] = content
    else:
        delta_kwargs["content"] = None
    delta_kwargs["tool_calls"] = tool_calls
    if reasoning_content is not None:
        delta_kwargs["reasoning_content"] = reasoning_content
    delta = SimpleNamespace(**delta_kwargs)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    chunk = SimpleNamespace(choices=[choice], usage=usage)
    return chunk


async def _fake_stream(chunks):
    """Async iterator wrapping a list of chunks."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_stream_response_text_only(open_router_provider):
    """Simple text streaming produces valid Anthropic SSE."""
    req = MockRequest()
    chunks = [
        _make_chunk(content="Hello"),
        _make_chunk(content=" world"),
        _make_chunk(finish_reason="stop"),
    ]
    stream = _fake_stream(chunks)

    with patch.object(
        open_router_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        events = [e async for e in open_router_provider.stream_response(req)]

    event_text = "".join(events)
    assert "message_start" in event_text
    assert "Hello" in event_text
    assert "world" in event_text
    assert "message_stop" in event_text


@pytest.mark.asyncio
async def test_stream_response_with_reasoning(open_router_provider):
    """Reasoning content is converted to Anthropic thinking blocks."""
    req = MockRequest()
    chunks = [
        _make_chunk(reasoning_content="Let me think..."),
        _make_chunk(content="Answer"),
        _make_chunk(finish_reason="stop"),
    ]
    stream = _fake_stream(chunks)

    with patch.object(
        open_router_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        events = [e async for e in open_router_provider.stream_response(req)]

    event_text = "".join(events)
    parsed = parse_sse_text(event_text)
    assert_anthropic_stream_contract(parsed)
    assert "Let me think..." in thinking_content(parsed)
    assert "Answer" in text_content(parsed)


@pytest.mark.asyncio
async def test_stream_response_suppresses_thinking_when_disabled(open_router_config):
    """When thinking is disabled, reasoning_content is suppressed."""
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = OpenRouterProvider(
            open_router_config.model_copy(update={"enable_thinking": False})
        )

    req = MockRequest()
    chunks = [
        _make_chunk(reasoning_content="secret thinking"),
        _make_chunk(content="Answer"),
        _make_chunk(finish_reason="stop"),
    ]
    stream = _fake_stream(chunks)

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        events = [e async for e in provider.stream_response(req)]

    event_text = "".join(events)
    assert "secret thinking" not in event_text
    assert "Answer" in event_text


@pytest.mark.asyncio
async def test_stream_response_tool_calls(open_router_provider):
    """Native tool calls are converted to Anthropic tool_use SSE blocks."""
    req = MockRequest()
    tool_call = SimpleNamespace(
        index=0,
        id="call_123",
        function=SimpleNamespace(
            name="Read",
            arguments='{"path": "/tmp/test"}',
        ),
    )
    chunks = [
        _make_chunk(content="Let me read that file."),
        _make_chunk(tool_calls=[tool_call]),
        _make_chunk(finish_reason="tool_calls"),
    ]
    stream = _fake_stream(chunks)

    with patch.object(
        open_router_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        events = [e async for e in open_router_provider.stream_response(req)]

    event_text = "".join(events)
    assert "tool_use" in event_text
    assert "Read" in event_text
    assert "message_stop" in event_text


@pytest.mark.asyncio
async def test_stream_response_error_path(open_router_provider):
    req = MockRequest()

    with patch.object(
        open_router_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=RuntimeError("API failed"),
    ):
        events = [e async for e in open_router_provider.stream_response(req)]

    event_text = "".join(events)
    assert "message_start" in event_text
    assert "API failed" in event_text
    assert "message_stop" in event_text


def test_default_base_url_is_opencode():
    """When no base_url is configured, the provider defaults to opencode.ai."""
    config = ProviderConfig(api_key="test_key")
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = OpenRouterProvider(config)
        assert provider._base_url == "https://opencode.ai/zen/v1"
