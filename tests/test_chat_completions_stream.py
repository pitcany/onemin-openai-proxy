import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

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


def test_stream_fallback_returns_valid_sse_response(client, mock_onemin_response):
    async def mock_post(*args, **kwargs):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = mock_onemin_response
        return mock_response

    async def mock_send(request, stream=False):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200

        async def aiter_bytes():
            yield b"Hello! "
            yield b"How can I help you?"

        mock_response.aiter_bytes = aiter_bytes
        mock_response.aclose = AsyncMock()
        mock_response.aread = AsyncMock(return_value=b"Hello! How can I help you?")
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.send = mock_send
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    content = response.text
    assert "data:" in content
    assert "[DONE]" in content


def test_stream_contains_chunk_structure(client, mock_onemin_response):
    async def mock_post(*args, **kwargs):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = mock_onemin_response
        return mock_response

    async def mock_send(request, stream=False):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200

        async def aiter_bytes():
            yield b"Hello!"

        mock_response.aiter_bytes = aiter_bytes
        mock_response.aclose = AsyncMock()
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mock_client.send = mock_send
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )

    content = response.text
    lines = [
        l for l in content.split("\n") if l.startswith("data:") and l != "data: [DONE]"
    ]

    assert len(lines) >= 1

    for line in lines:
        json_str = line.replace("data: ", "")
        if json_str.strip():
            chunk = json.loads(json_str)
            assert "id" in chunk
            assert chunk["object"] == "chat.completion.chunk"
            assert "created" in chunk
            assert "model" in chunk
            assert "choices" in chunk
            assert len(chunk["choices"]) == 1
            assert "delta" in chunk["choices"][0]


def test_stream_includes_request_id_header(client, mock_onemin_response):
    async def mock_send(request, stream=False):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200

        async def aiter_bytes():
            yield b"Hello!"

        mock_response.aiter_bytes = aiter_bytes
        mock_response.aclose = AsyncMock()
        return mock_response

    with patch("app.onemin_client.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.send = mock_send
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
            headers={"X-Request-ID": "test-req-123"},
        )

    assert response.headers.get("x-request-id") == "test-req-123"


def test_models_endpoint_returns_default_model(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1

    model_ids = [m["id"] for m in data["data"]]
    assert "gpt-4o-mini" in model_ids

    for model in data["data"]:
        assert model["object"] == "model"
        assert model["owned_by"] == "1minai"


def test_models_endpoint_includes_extra_models():
    os.environ["EXTRA_MODELS"] = "gpt-4o,claude-3-5-sonnet"

    from importlib import reload
    import app.settings as settings_module

    reload(settings_module)
    settings_module.get_settings.cache_clear()

    import app.main as main_module

    reload(main_module)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as test_client:
        response = test_client.get("/v1/models")

    del os.environ["EXTRA_MODELS"]
    reload(settings_module)
    settings_module.get_settings.cache_clear()

    assert response.status_code == 200
    data = response.json()
    model_ids = [m["id"] for m in data["data"]]

    assert "gpt-4o-mini" in model_ids
    assert "gpt-4o" in model_ids
    assert "claude-3-5-sonnet" in model_ids
