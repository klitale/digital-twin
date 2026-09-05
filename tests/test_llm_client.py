from __future__ import annotations

import pytest
from pydantic import SecretStr

from fake_openai_server import FakeOpenAIServer
from twin.core.llm_client import LLMClient, LLMError, reasoning_off_params

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "привет"}]


def test_chat_returns_text_usage_and_sends_key() -> None:
    with FakeOpenAIServer(reply="привет!") as server:
        client = LLMClient(server.base_url, SecretStr("sk-test"), model="anthropic/x")
        result = client.chat(MESSAGES, temperature=0.2, max_tokens=50)
    assert result.text == "привет!"
    assert result.model == "anthropic/x"
    assert result.prompt_tokens and result.completion_tokens
    assert result.latency_ms >= 0 and result.finish_reason == "stop"
    body = server.requests[0]
    assert body["model"] == "anthropic/x" and body["temperature"] == 0.2
    assert body["max_tokens"] == 50 and body["messages"] == MESSAGES
    assert server.auth_headers == ["Bearer sk-test"]
    assert "sk-test" not in repr(client)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek/deepseek-v4-flash", {"thinking": {"type": "disabled"}}),
        ("dashscope/qwen3.5-flash", {"enable_thinking": False}),
        ("openai/gpt-5.4-mini", {"reasoning_effort": "none"}),
        ("anthropic/claude-sonnet-4-6", {}),
    ],
)
def test_reasoning_off_params_reach_the_request(model: str, expected: dict[str, object]) -> None:
    assert reasoning_off_params(model) == expected
    with FakeOpenAIServer() as server:
        LLMClient(server.base_url, "k", model=model).chat(MESSAGES)
    body = server.requests[0]
    for key, value in expected.items():
        assert body[key] == value
    if not expected:
        assert "thinking" not in body and "reasoning_effort" not in body


def test_reasoning_off_can_be_disabled_and_extra_body_merged() -> None:
    with FakeOpenAIServer() as server:
        client = LLMClient(server.base_url, "k", model="deepseek/x", reasoning_off=False)
        client.chat(MESSAGES, extra_body={"top_p": 0.5})
    assert "thinking" not in server.requests[0] and server.requests[0]["top_p"] == 0.5


def test_retries_then_succeeds() -> None:
    with FakeOpenAIServer(reply="ок", fail_times=1) as server:
        client = LLMClient(server.base_url, "k", model="m", max_retries=2)
        assert client.chat(MESSAGES).text == "ок"
    assert len(server.requests) == 2


def test_gives_up_after_bounded_retries() -> None:
    with FakeOpenAIServer(fail_times=10, fail_status=503) as server:
        client = LLMClient(server.base_url, "k", model="m", max_retries=1)
        with pytest.raises(LLMError):
            client.chat(MESSAGES)
    assert len(server.requests) == 2  # one attempt + one retry


def test_empty_completion_is_an_error() -> None:
    with FakeOpenAIServer(reply="   ") as server, pytest.raises(LLMError, match="empty"):
        LLMClient(server.base_url, "k", model="m").chat(MESSAGES)
