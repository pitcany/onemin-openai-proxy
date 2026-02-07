"""
Translation layer between OpenAI API format and 1minAI API format.
All mapping logic is centralized here for easy maintenance.
"""

import logging
import time
import uuid
from typing import TYPE_CHECKING, Optional, Union

from .onemin_schemas import ChatPromptObject, OneMinAIChatRequest
from .openai_schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    ContentType,
    UsageInfo,
)
from .settings import Settings

if TYPE_CHECKING:
    from .onemin_client import OneMinAIClient

logger = logging.getLogger(__name__)


def extract_text_from_content(content: ContentType) -> str:
    """
    Extract text from content that may be string or multimodal array.

    Args:
        content: String or list of content parts

    Returns:
        Extracted text content
    """
    if isinstance(content, str):
        return content

    # Multimodal content - extract text parts only
    text_parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        elif hasattr(part, "type") and part.type == "text":
            text_parts.append(part.text)
    return " ".join(text_parts)


def extract_image_urls(content: ContentType) -> list[str]:
    """
    Extract image URLs from multimodal content.

    Args:
        content: String or list of content parts

    Returns:
        List of image URLs found
    """
    if isinstance(content, str):
        return []

    urls = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "image_url":
                image_url = part.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                    if url:
                        urls.append(url)
        elif hasattr(part, "type") and part.type == "image_url":
            if hasattr(part, "image_url") and isinstance(part.image_url, dict):
                url = part.image_url.get("url", "")
                if url:
                    urls.append(url)
    return urls


def detect_pdf_urls(content: ContentType) -> list[str]:
    """Extract PDF URLs from content."""
    if isinstance(content, str):
        import re

        # Simple PDF URL detection
        pdf_pattern = r"https?://[^\s]+\.pdf(?:\?[^\s]*)?"
        return re.findall(pdf_pattern, content, re.IGNORECASE)
    return []


def detect_youtube_url(content: ContentType) -> Optional[str]:
    """Extract YouTube URL from content."""
    if isinstance(content, str):
        import re

        # YouTube URL patterns
        patterns = [
            r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
            r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(0)
    return None


def detect_feature_type(messages: list) -> str:
    """
    Detect the appropriate 1minAI feature type based on message content.
    Priority: PDF > YouTube > Image > AI
    """
    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                # Check for PDF URLs
                if detect_pdf_urls(msg.content):
                    return "CHAT_WITH_PDF"
                # Check for YouTube URLs
                if detect_youtube_url(msg.content):
                    return "CHAT_WITH_YOUTUBE_VIDEO"
            else:
                # Multimodal content - check for images
                image_urls = extract_image_urls(msg.content)
                if image_urls:
                    return "CHAT_WITH_IMAGE"

    return "CHAT_WITH_AI"


def messages_to_prompt(messages: list) -> str:
    """
    Convert OpenAI-style messages array to a single prompt string.

    Format:
    - System messages at top, prefixed with "System:"
    - Then conversation turns: "User: ...\nAssistant: ..."
    - Last user message is always included

    Args:
        messages: List of ChatMessage objects

    Returns:
        Combined prompt string
    """
    if not messages:
        return ""

    parts: list[str] = []

    # First, collect all system messages
    system_messages = [m for m in messages if m.role == "system"]
    for msg in system_messages:
        text = extract_text_from_content(msg.content)
        parts.append(f"System: {text}")

    # Then add conversation in order (excluding system messages)
    for msg in messages:
        if msg.role == "system":
            continue
        text = extract_text_from_content(msg.content)
        if msg.role == "user":
            parts.append(f"User: {text}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {text}")
        elif msg.role == "tool":
            # Tool messages are treated like system context
            parts.append(f"Tool: {text}")

    return "\n".join(parts)


async def openai_to_onemin_request(
    request: ChatCompletionRequest,
    settings: Settings,
    onemin_client: Optional["OneMinAIClient"] = None,
) -> OneMinAIChatRequest:
    """
    Translate OpenAI chat completion request to 1minAI request format.
    Async because CHAT_WITH_IMAGE requires downloading and uploading images.
    """
    model = request.model if request.model else settings.default_1min_model
    prompt = messages_to_prompt(request.messages)
    feature_type = detect_feature_type(request.messages)

    image_list: Optional[list[str]] = None
    conversation_id: Optional[str] = None

    if feature_type == "CHAT_WITH_PDF" and onemin_client:
        # Extract PDF URLs and upload them
        pdf_urls = []
        for msg in request.messages:
            if msg.role == "user" and isinstance(msg.content, str):
                pdf_urls.extend(detect_pdf_urls(msg.content))

        if pdf_urls:
            file_list = []
            for pdf_url in pdf_urls:
                try:
                    (
                        pdf_bytes,
                        filename,
                        content_type,
                    ) = await onemin_client.download_image(pdf_url)
                    asset_key = await onemin_client.upload_asset(
                        pdf_bytes, filename, content_type
                    )
                    file_list.append(asset_key)
                    logger.info(f"Uploaded PDF to 1minAI asset: {asset_key}")
                except Exception as e:
                    logger.error(f"Failed to process PDF {pdf_url}: {e}")

            if file_list:
                conversation_id = await onemin_client.create_conversation(
                    feature_type="CHAT_WITH_PDF",
                    model=model,
                    title="PDF Chat",
                    file_list=file_list,
                )
                logger.info(f"Created conversation for PDF: {conversation_id}")
            else:
                feature_type = "CHAT_WITH_AI"

    elif feature_type == "CHAT_WITH_YOUTUBE_VIDEO" and onemin_client:
        # Extract YouTube URL
        youtube_url = None
        for msg in request.messages:
            if msg.role == "user" and isinstance(msg.content, str):
                youtube_url = detect_youtube_url(msg.content)
                if youtube_url:
                    break

        if youtube_url:
            try:
                conversation_id = await onemin_client.create_conversation(
                    feature_type="CHAT_WITH_YOUTUBE_VIDEO",
                    model=model,
                    title="YouTube Chat",
                    youtube_url=youtube_url,
                )
                logger.info(f"Created conversation for YouTube: {conversation_id}")
            except Exception as e:
                logger.error(f"Failed to create YouTube conversation: {e}")
                feature_type = "CHAT_WITH_AI"
        else:
            feature_type = "CHAT_WITH_AI"

    elif feature_type == "CHAT_WITH_IMAGE" and onemin_client:
        # Collect all image URLs from all user messages
        all_image_urls: list[str] = []
        for msg in request.messages:
            if msg.role == "user":
                all_image_urls.extend(extract_image_urls(msg.content))

        if all_image_urls:
            image_list = []
            for url in all_image_urls:
                try:
                    (
                        img_bytes,
                        filename,
                        content_type,
                    ) = await onemin_client.download_image(url)
                    asset_key = await onemin_client.upload_asset(
                        img_bytes, filename, content_type
                    )
                    image_list.append(asset_key)
                    logger.info(f"Uploaded image to 1minAI asset: {asset_key}")
                except Exception as e:
                    logger.error(f"Failed to process image {url}: {e}")

            if not image_list:
                # All uploads failed — fall back to text-only
                feature_type = "CHAT_WITH_AI"
                image_list = None

    prompt_object = ChatPromptObject(
        prompt=prompt,
        isMixed=False,
        imageList=image_list,
        webSearch=settings.default_websearch,
        numOfSite=settings.default_num_of_site,
        maxWord=settings.default_max_word,
    )

    return OneMinAIChatRequest(
        type=feature_type,
        model=model,
        promptObject=prompt_object,
        conversationId=conversation_id,
    )


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using word-based heuristic.

    Uses ~1.3 tokens per word as a rough approximation.
    This is NOT accurate but provides a reasonable estimate.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    words = text.split()
    return int(len(words) * 1.3)


def onemin_to_openai_response(
    onemin_response: dict,
    model: str,
    prompt_text: str,
) -> tuple[ChatCompletionResponse, bool]:
    """
    Translate 1minAI response to OpenAI chat completion response.

    Args:
        onemin_response: Raw response from 1minAI
        model: Model name used
        prompt_text: Original prompt text for token estimation

    Returns:
        Tuple of (ChatCompletionResponse, usage_estimated: bool)
    """
    # Extract assistant response text
    assistant_text = ""
    ai_record = onemin_response.get("aiRecord", {})

    # Try to get result from aiRecordDetail.resultObject
    detail = ai_record.get("aiRecordDetail", {})
    result_object = detail.get("resultObject")

    if result_object:
        if isinstance(result_object, list) and result_object:
            # For chat, resultObject is typically a list with the response text
            assistant_text = str(result_object[0]) if result_object else ""
        elif isinstance(result_object, str):
            assistant_text = result_object
    else:
        # Fallback: check for direct text response
        assistant_text = ai_record.get("response", "")

    # Generate completion ID
    completion_id = f"chatcmpl_{uuid.uuid4().hex[:24]}"

    # Estimate token usage (1minAI doesn't provide this for chat)
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(assistant_text)
    usage_estimated = True

    # Determine finish reason based on status
    status = ai_record.get("status", "SUCCESS")
    finish_reason: Optional[str] = "stop"
    if status == "FAILED":
        finish_reason = None
    elif status == "CONTENT_FILTER":
        finish_reason = "content_filter"

    response = ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessageResponse(
                    role="assistant",
                    content=assistant_text,
                ),
                finish_reason=finish_reason,  # type: ignore
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response, usage_estimated


def create_stream_chunk(
    chunk_id: str,
    model: str,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    include_role: bool = False,
) -> dict:
    """
    Create an OpenAI-style streaming chunk.

    Args:
        chunk_id: Unique ID for the stream
        model: Model name
        content: Content delta (optional)
        finish_reason: Finish reason (optional)
        include_role: Whether to include role in delta

    Returns:
        Chunk dictionary
    """
    delta: dict = {}
    if include_role:
        delta["role"] = "assistant"
    if content is not None:
        delta["content"] = content

    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
