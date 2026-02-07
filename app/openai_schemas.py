"""
OpenAI-compatible request and response schemas.
These match the OpenAI API format for client compatibility.
"""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class ContentPartText(BaseModel):
    """Text content part for multimodal messages."""

    type: Literal["text"] = "text"
    text: str


class ContentPartImageUrl(BaseModel):
    """Image URL content part for multimodal messages."""

    type: Literal["image_url"] = "image_url"
    image_url: dict  # {"url": "..."} or {"url": "...", "detail": "..."}


# Content can be a string or array of content parts (multimodal)
ContentType = Union[str, list[Union[ContentPartText, ContentPartImageUrl, dict]]]


class ChatMessage(BaseModel):
    """A single message in the chat conversation."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="The role of the message author"
    )
    content: ContentType = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    """OpenAI-style chat completion request."""

    model: Optional[str] = Field(
        default=None,
        description="Model to use. If not provided, uses DEFAULT_1MIN_MODEL",
    )
    messages: list[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    max_tokens: Optional[int] = Field(
        default=None, gt=0, description="Maximum tokens to generate"
    )
    stream: Optional[bool] = Field(
        default=False, description="Whether to stream the response"
    )
    stop: Optional[Union[str, list[str]]] = Field(
        default=None, description="Stop sequences"
    )


class ChatMessageResponse(BaseModel):
    """Response message from the assistant."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: ChatMessageResponse
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = "stop"


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-style chat completion response."""

    id: str = Field(..., description="Unique identifier for the completion")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: list[ChatChoice]
    usage: UsageInfo


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streaming response."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list["ChatChoiceDelta"]


class ChatChoiceDelta(BaseModel):
    """Delta for streaming responses."""

    index: int = 0
    delta: "DeltaMessage"
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class DeltaMessage(BaseModel):
    """Partial message in streaming response."""

    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "1minai"


class ModelList(BaseModel):
    """Response for /v1/models endpoint."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response for /healthz endpoint."""

    ok: bool = True


# Update forward references
ChatCompletionChunk.model_rebuild()
ChatChoiceDelta.model_rebuild()
