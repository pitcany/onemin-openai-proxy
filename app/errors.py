"""
OpenAI-style error response helpers.
Translates upstream errors into OpenAI-compatible error format.
"""

import re
from typing import Literal, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class OpenAIErrorDetail(BaseModel):
    """OpenAI-style error detail structure."""

    message: str
    type: Literal[
        "upstream_error",
        "invalid_request_error",
        "authentication_error",
        "rate_limit_error",
    ]
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIError(BaseModel):
    """OpenAI-style error response wrapper."""

    error: OpenAIErrorDetail


def create_error_response(
    status_code: int,
    message: str,
    error_type: str = "upstream_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    """Create an OpenAI-style error response."""
    # Sanitize message to remove any potential API keys
    sanitized_message = sanitize_message(message)

    error = OpenAIError(
        error=OpenAIErrorDetail(
            message=sanitized_message,
            type=error_type,  # type: ignore
            param=param,
            code=code,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(),
    )


def sanitize_message(message: str) -> str:
    """Remove potential API keys and sensitive data from error messages."""
    # Pattern to match API key-like strings (alphanumeric, 20+ chars)
    key_pattern = r"\b[a-zA-Z0-9_-]{20,}\b"
    sanitized = re.sub(key_pattern, "[REDACTED]", message)

    # Also redact common key patterns
    sanitized = re.sub(
        r"API-KEY:\s*\S+", "API-KEY: [REDACTED]", sanitized, flags=re.IGNORECASE
    )
    sanitized = re.sub(
        r'api[_-]?key["\']?\s*[:=]\s*["\']?\S+',
        "api_key: [REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )

    return sanitized


def map_status_to_error_type(status_code: int) -> str:
    """Map HTTP status code to OpenAI error type."""
    if status_code in (401, 403):
        return "authentication_error"
    elif status_code == 429:
        return "rate_limit_error"
    elif 400 <= status_code < 500:
        return "invalid_request_error"
    else:
        return "upstream_error"


def map_status_code(upstream_status: int) -> int:
    """Map upstream status code to proxy response status code."""
    if upstream_status in (401, 403):
        return 401
    elif upstream_status == 429:
        return 429
    elif 400 <= upstream_status < 500:
        return upstream_status
    elif upstream_status >= 500:
        return 502
    else:
        return upstream_status


class ProxyHTTPException(HTTPException):
    """Custom exception for proxy errors with OpenAI-style formatting."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: Optional[str] = None,
        param: Optional[str] = None,
        code: Optional[str] = None,
    ):
        self.error_type = error_type or map_status_to_error_type(status_code)
        self.param = param
        self.code = code
        self.message = sanitize_message(message)
        super().__init__(status_code=status_code, detail=self.message)
