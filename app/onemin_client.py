"""
HTTP client for 1minAI API with retry logic, timeouts, and streaming support.
"""

import asyncio
import json
import logging
import random
from typing import AsyncIterator, Optional

import httpx

from .errors import ProxyHTTPException, map_status_code, map_status_to_error_type
from .onemin_schemas import OneMinAIChatRequest
from .settings import Settings

logger = logging.getLogger(__name__)


class OneMinAIClient:
    """Async HTTP client for 1minAI API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.one_min_ai_base_url.rstrip("/")
        self.timeout = httpx.Timeout(
            timeout=float(settings.request_timeout_secs),
            connect=10.0,
        )
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "API-KEY": self.settings.one_min_ai_api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _get_endpoint(self, streaming: bool = False) -> str:
        """Get the appropriate endpoint URL."""
        base = f"{self.base_url}/api/features"
        if streaming:
            return f"{base}?isStreaming=true"
        return base

    async def _make_request_with_retry(
        self,
        request: OneMinAIChatRequest,
        streaming: bool = False,
    ) -> httpx.Response:
        """
        Make a request with exponential backoff retry for rate limits.

        Args:
            request: The 1minAI request
            streaming: Whether to use streaming endpoint

        Returns:
            httpx.Response

        Raises:
            ProxyHTTPException: On error after retries exhausted
        """
        client = await self._get_client()
        endpoint = self._get_endpoint(streaming)
        payload = request.model_dump(exclude_none=True)

        last_exception: Optional[Exception] = None
        retries = self.settings.retries

        for attempt in range(retries + 1):
            try:
                if streaming:
                    # For streaming, we don't read the response here
                    response = await client.send(
                        client.build_request(
                            "POST",
                            endpoint,
                            json=payload,
                        ),
                        stream=True,
                    )
                else:
                    response = await client.post(endpoint, json=payload)

                # Check for rate limit
                if response.status_code == 429:
                    if attempt < retries:
                        # Exponential backoff with jitter
                        base_delay = 2**attempt
                        jitter = random.uniform(0, 1)
                        delay = base_delay + jitter
                        logger.warning(
                            f"Rate limited (attempt {attempt + 1}/{retries + 1}), "
                            f"retrying in {delay:.2f}s"
                        )
                        if streaming:
                            await response.aclose()
                        await asyncio.sleep(delay)
                        continue
                    else:
                        if streaming:
                            await response.aclose()
                        raise ProxyHTTPException(
                            status_code=429,
                            message="Rate limit exceeded after retries",
                            error_type="rate_limit_error",
                        )

                # Handle other error status codes
                if response.status_code >= 400:
                    if streaming:
                        body = await response.aread()
                        await response.aclose()
                        error_text = body.decode("utf-8", errors="replace")
                    else:
                        error_text = response.text

                    mapped_status = map_status_code(response.status_code)
                    error_type = map_status_to_error_type(response.status_code)

                    raise ProxyHTTPException(
                        status_code=mapped_status,
                        message=f"1minAI API error: {error_text[:500]}",
                        error_type=error_type,
                    )

                return response

            except httpx.TimeoutException as e:
                last_exception = e
                logger.error(f"Request timeout: {e}")
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                raise ProxyHTTPException(
                    status_code=504,
                    message="Request to upstream API timed out",
                    error_type="upstream_error",
                )

            except httpx.RequestError as e:
                last_exception = e
                logger.error(f"Request error: {e}")
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                raise ProxyHTTPException(
                    status_code=502,
                    message=f"Failed to connect to upstream API: {str(e)}",
                    error_type="upstream_error",
                )

            except ProxyHTTPException:
                raise

            except Exception as e:
                last_exception = e
                logger.exception(f"Unexpected error: {e}")
                raise ProxyHTTPException(
                    status_code=500,
                    message=f"Internal proxy error: {str(e)}",
                    error_type="upstream_error",
                )

        # Should not reach here, but just in case
        raise ProxyHTTPException(
            status_code=500,
            message=f"Request failed after {retries + 1} attempts: {last_exception}",
            error_type="upstream_error",
        )

    async def upload_asset(
        self, file_bytes: bytes, filename: str, content_type: str
    ) -> str:
        """Upload a file to 1minAI Asset API, return the asset key."""
        url = f"{self.base_url}/api/assets"

        # Create a separate client for file upload (avoid JSON Content-Type)
        async with httpx.AsyncClient(timeout=self.timeout) as upload_client:
            response = await upload_client.post(
                url,
                files={"asset": (filename, file_bytes, content_type)},
                headers={"API-KEY": self.settings.one_min_ai_api_key},
            )

            if response.status_code >= 400:
                raise ProxyHTTPException(
                    status_code=response.status_code,
                    message=f"Asset upload failed: {response.text[:500]}",
                    error_type="upstream_error",
                )

            data = response.json()
            return data["fileContent"]["uuid"]

    async def download_file(self, url: str) -> tuple[bytes, str, str]:
        """Download file from URL, return (bytes, filename, content_type)."""
        client = await self._get_client()
        response = await client.get(url, headers={"User-Agent": "1minAI-Proxy/1.0"})

        if response.status_code >= 400:
            raise ProxyHTTPException(
                status_code=400,
                message=f"Failed to download file from {url}",
                error_type="invalid_request_error",
            )

        content_type = response.headers.get("content-type", "application/octet-stream")
        content_type = content_type.split(";")[0].strip()

        ext_map = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
            "image/gif": "gif",
            "application/pdf": "pdf",
        }
        ext = ext_map.get(content_type, url.split(".")[-1].split("?")[0] or "bin")
        filename = f"upload.{ext}"

        return response.content, filename, content_type

    async def download_image(self, url: str) -> tuple[bytes, str, str]:
        """Alias for download_file for backward compatibility."""
        return await self.download_file(url)

    async def create_conversation(
        self,
        feature_type: str,
        model: str,
        title: str = "Conversation",
        file_list: Optional[list[str]] = None,
        youtube_url: Optional[str] = None,
    ) -> str:
        """Create a conversation and return the conversation ID."""
        client = await self._get_client()
        url = f"{self.base_url}/api/conversations"

        payload = {"type": feature_type, "model": model, "title": title}

        if file_list:
            payload["fileList"] = file_list
        if youtube_url:
            payload["youtubeUrl"] = youtube_url

        response = await client.post(url, json=payload)

        if response.status_code >= 400:
            raise ProxyHTTPException(
                status_code=response.status_code,
                message=f"Conversation creation failed: {response.text[:500]}",
                error_type="upstream_error",
            )

        data = response.json()
        return data["conversation"]["uuid"]

    async def chat_completion(
        self,
        request: OneMinAIChatRequest,
    ) -> dict:
        """
        Make a non-streaming chat completion request.

        Args:
            request: The 1minAI chat request

        Returns:
            Response JSON as dict
        """
        response = await self._make_request_with_retry(request, streaming=False)
        return response.json()

    async def chat_completion_stream(
        self,
        request: OneMinAIChatRequest,
    ) -> AsyncIterator[str]:
        """
        Make a streaming chat completion request.

        Yields text chunks from the streaming response.
        Falls back to non-streaming if streaming format is not parseable.

        Args:
            request: The 1minAI chat request

        Yields:
            Text chunks from the response
        """
        response = await self._make_request_with_retry(request, streaming=True)

        try:
            # Buffer for accumulating partial lines
            buffer = ""

            async for chunk in response.aiter_bytes():
                text = chunk.decode("utf-8", errors="replace")
                buffer += text

                # Try to yield complete lines/chunks
                while buffer:
                    # Check for SSE format (data: ...)
                    if buffer.startswith("data:"):
                        newline_idx = buffer.find("\n")
                        if newline_idx == -1:
                            break  # Incomplete line, wait for more
                        line = buffer[:newline_idx]
                        buffer = buffer[newline_idx + 1 :].lstrip("\n")

                        data = line[5:].strip()  # Remove "data:" prefix
                        if data and data != "[DONE]":
                            try:
                                # Try to parse as JSON and extract content
                                parsed = json.loads(data)
                                if isinstance(parsed, dict):
                                    # 1minAI might have different stream format
                                    content = parsed.get(
                                        "content", parsed.get("text", "")
                                    )
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                # Not JSON, yield as plain text
                                yield data
                    else:
                        # Plain text streaming - yield character by character or chunks
                        # Find a good break point
                        if "\n" in buffer:
                            idx = buffer.find("\n")
                            yield buffer[: idx + 1]
                            buffer = buffer[idx + 1 :]
                        elif len(buffer) > 100:
                            # Yield in chunks if buffer gets too large
                            yield buffer[:50]
                            buffer = buffer[50:]
                        else:
                            # Small buffer, might be partial - yield it
                            yield buffer
                            buffer = ""
                            break

            # Yield any remaining buffer
            if buffer.strip():
                yield buffer

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise
        finally:
            await response.aclose()

    async def chat_completion_stream_fallback(
        self,
        request: OneMinAIChatRequest,
    ) -> tuple[dict, bool]:
        """
        Fallback method when streaming is not supported or fails.

        Makes a non-streaming request and returns the result.

        Args:
            request: The 1minAI chat request

        Returns:
            Tuple of (response dict, is_fallback: True)
        """
        response = await self.chat_completion(request)
        return response, True
