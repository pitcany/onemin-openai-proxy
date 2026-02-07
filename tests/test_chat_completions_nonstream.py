import json
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def setup_env():
    os.environ["ONE_MIN_AI_API_KEY"] = "test-api-key"
    os.environ["DEFAULT_1MIN_MODEL"] = "gpt-4o-mini"
    yield


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_onemin_response():
    return {
        "aiRecord": {
            "uuid": "test-uuid-123",
            "model": "gpt-4o-mini",
            "type": "CHAT_WITH_AI",
            "status": "SUCCESS",
            "aiRecordDetail": {
                "promptObject": {"prompt": "User: Hello"},
                "resultObject": ["Hello! How can I help you today?"],
            },
        }
    }


def test_chat_completions_maps_to_onemin_payload(client, mock_onemin_response):
    captured_request = {}

    async def mock_post(*args, **kwargs):
        captured_request["json"] = kwargs.get("json")
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = mock_onemin_response
        mock_response.text = json.dumps(mock_onemin_response)
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200

    assert captured_request.get("json") is not None
    payload = captured_request["json"]
    assert payload["type"] == "CHAT_WITH_AI"
    assert payload["model"] == "gpt-4o-mini"
    assert "promptObject" in payload
    assert "prompt" in payload["promptObject"]
    assert "User: Hello" in payload["promptObject"]["prompt"]


def test_chat_completions_normalizes_response_to_openai_shape(
    client, mock_onemin_response
):
    async def mock_post(*args, **kwargs):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = mock_onemin_response
        mock_response.text = json.dumps(mock_onemin_response)
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl_")
    assert "created" in data
    assert "model" in data
    assert "choices" in data
    assert len(data["choices"]) == 1

    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello! How can I help you today?"
    assert choice["finish_reason"] == "stop"

    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]


def test_chat_completions_uses_default_model_when_not_specified(
    client, mock_onemin_response
):
    captured_request = {}

    async def mock_post(*args, **kwargs):
        captured_request["json"] = kwargs.get("json")
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = mock_onemin_response
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    assert captured_request["json"]["model"] == "gpt-4o-mini"


def test_chat_completions_handles_system_messages(client, mock_onemin_response):
    captured_request = {}

    async def mock_post(*args, **kwargs):
        captured_request["json"] = kwargs.get("json")
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = mock_onemin_response
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ]
            },
        )

    assert response.status_code == 200
    prompt = captured_request["json"]["promptObject"]["prompt"]
    assert "System: You are a helpful assistant." in prompt
    assert "User: Hello" in prompt


def test_chat_completions_401_returns_authentication_error(client):
    async def mock_post(*args, **kwargs):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "authentication_error"


def test_chat_completions_429_returns_rate_limit_error_after_retries(client):
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.text = "Rate limited"
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

    assert response.status_code == 429
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "rate_limit_error"
    assert call_count == 3


def test_chat_completions_5xx_returns_upstream_error(client):
    async def mock_post(*args, **kwargs):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 502
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "upstream_error"


def test_chat_completions_includes_usage_estimated_header(client, mock_onemin_response):
    async def mock_post(*args, **kwargs):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = mock_onemin_response
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    assert response.headers.get("x-proxy-usage") == "estimated"
