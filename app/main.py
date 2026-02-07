"""
FastAPI application entry point with OpenAI-compatible endpoints.
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .errors import ProxyHTTPException, create_error_response
from .onemin_client import OneMinAIClient
from .openai_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    ModelInfo,
    ModelList,
)
from .settings import Settings, get_settings
from .translate import (
    create_stream_chunk,
    messages_to_prompt,
    onemin_to_openai_response,
    openai_to_onemin_request,
)

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global client instance
client: OneMinAIClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    global client
    settings = get_settings()
    client = OneMinAIClient(settings)
    logger.info(f"1minAI proxy starting on {settings.proxy_host}:{settings.proxy_port}")
    yield
    await client.close()
    logger.info("1minAI proxy shutting down")


app = FastAPI(
    title="1minAI OpenAI-Compatible Proxy",
    description="Proxies OpenAI API requests to 1minAI's AI Feature API",
    version="1.0.0",
    lifespan=lifespan,
)


def get_request_id(x_request_id: str | None = Header(None)) -> str:
    """Get or generate a request correlation ID."""
    return x_request_id or f"req_{uuid.uuid4().hex[:16]}"


def log_request(
    request_id: str,
    model: str | None,
    message_count: int,
    prompt_length: int,
) -> None:
    """Log request details if logging is enabled (with redaction)."""
    if settings.enable_request_logging:
        logger.info(
            f"[{request_id}] model={model or 'default'}, "
            f"messages={message_count}, "
            f"prompt_chars={prompt_length}"
        )


@app.exception_handler(ProxyHTTPException)
async def proxy_exception_handler(
    request: Request, exc: ProxyHTTPException
) -> JSONResponse:
    """Handle ProxyHTTPException and return OpenAI-style error."""
    return create_error_response(
        status_code=exc.status_code,
        message=exc.message,
        error_type=exc.error_type,
        param=exc.param,
        code=exc.code,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return create_error_response(
        status_code=500,
        message="Internal server error",
        error_type="upstream_error",
    )


@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def healthz() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(ok=True)


@app.get("/v1/models", response_model=ModelList, tags=["Models"])
async def list_models() -> ModelList:
    """List available models."""
    settings = get_settings()

    models = [
        ModelInfo(id=settings.default_1min_model, owned_by="1minai"),
    ]

    # Add extra models if configured
    for model_name in settings.get_extra_models_list():
        if model_name != settings.default_1min_model:
            models.append(ModelInfo(id=model_name, owned_by="1minai"))

    return ModelList(data=models)


@app.post("/v1/chat/completions", response_model=None, tags=["Chat"])
async def chat_completions(
    request: ChatCompletionRequest,
    x_request_id: str | None = Header(None, alias="X-Request-ID"),
) -> JSONResponse | StreamingResponse:
    settings = get_settings()
    request_id = get_request_id(x_request_id)

    # Convert messages to prompt for logging
    prompt_text = messages_to_prompt(request.messages)

    log_request(
        request_id=request_id,
        model=request.model,
        message_count=len(request.messages),
        prompt_length=len(prompt_text),
    )

    onemin_request = await openai_to_onemin_request(
        request, settings, onemin_client=client
    )
    model_used = onemin_request.model

    if request.stream:
        return await _handle_streaming(
            onemin_request=onemin_request,
            model=model_used,
            prompt_text=prompt_text,
            request_id=request_id,
        )
    else:
        return await _handle_non_streaming(
            onemin_request=onemin_request,
            model=model_used,
            prompt_text=prompt_text,
            request_id=request_id,
        )


async def _handle_non_streaming(
    onemin_request,
    model: str,
    prompt_text: str,
    request_id: str,
) -> JSONResponse:
    """Handle non-streaming chat completion."""
    response_data = await client.chat_completion(onemin_request)

    openai_response, usage_estimated = onemin_to_openai_response(
        response_data,
        model=model,
        prompt_text=prompt_text,
    )

    headers = {"X-Request-ID": request_id}
    if usage_estimated:
        headers["X-Proxy-Usage"] = "estimated"

    return JSONResponse(
        content=openai_response.model_dump(),
        headers=headers,
    )


async def _handle_streaming(
    onemin_request,
    model: str,
    prompt_text: str,
    request_id: str,
) -> StreamingResponse:
    """Handle streaming chat completion."""

    async def generate() -> AsyncIterator[bytes]:
        chunk_id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
        streaming_worked = False

        try:
            # Try streaming first
            first_chunk = True
            async for text_chunk in client.chat_completion_stream(onemin_request):
                streaming_worked = True
                if first_chunk:
                    # First chunk includes role
                    chunk = create_stream_chunk(
                        chunk_id=chunk_id,
                        model=model,
                        content=text_chunk,
                        include_role=True,
                    )
                    first_chunk = False
                else:
                    chunk = create_stream_chunk(
                        chunk_id=chunk_id,
                        model=model,
                        content=text_chunk,
                    )
                yield f"data: {json.dumps(chunk)}\n\n".encode()

            # Send final chunk with finish_reason
            final_chunk = create_stream_chunk(
                chunk_id=chunk_id,
                model=model,
                finish_reason="stop",
            )
            yield f"data: {json.dumps(final_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.warning(f"Streaming failed, falling back to non-streaming: {e}")

            if not streaming_worked:
                # Fall back to non-streaming
                try:
                    response_data, _ = await client.chat_completion_stream_fallback(
                        onemin_request
                    )
                    openai_response, _ = onemin_to_openai_response(
                        response_data,
                        model=model,
                        prompt_text=prompt_text,
                    )

                    # Emit as a single streamed response
                    content = openai_response.choices[0].message.content

                    # First chunk with role
                    chunk = create_stream_chunk(
                        chunk_id=chunk_id,
                        model=model,
                        content=content,
                        include_role=True,
                    )
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                    # Final chunk
                    final_chunk = create_stream_chunk(
                        chunk_id=chunk_id,
                        model=model,
                        finish_reason="stop",
                    )
                    yield f"data: {json.dumps(final_chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"

                except Exception as fallback_e:
                    logger.error(f"Fallback also failed: {fallback_e}")
                    error_chunk = {
                        "error": {
                            "message": str(fallback_e)[:200],
                            "type": "upstream_error",
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"

    headers = {
        "X-Request-ID": request_id,
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.proxy_host,
        port=settings.proxy_port,
        reload=True,
    )
