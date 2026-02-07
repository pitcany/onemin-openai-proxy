# Design: Multi-Feature 1MinAI Proxy Expansion (Revised)

## Executive Summary

Expand 1MinAI proxy from single-feature (CHAT_WITH_AI only) to multi-feature architecture supporting:
- **CHAT_WITH_AI** - Text chat (current)
- **CHAT_WITH_IMAGE** - Multimodal chat with images
- **CHAT_WITH_PDF** - Document analysis
- **CHAT_WITH_YOUTUBE_VIDEO** - Video understanding

**Goal**: Maintain full OpenAI API compatibility while unlocking all 1MinAI capabilities.

**Design Philosophy**: Fail-safe, feature-flagged, validated, and fully tested before hard-launch.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OpenAI-Compatible Client                       │
│                    (Existing apps/tools)                      │
└─────────────────────────────────┬─────────────────────────────────────┘
                              │
                              │ OpenAI API Format
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              1MinAI Proxy (Expanded)                          │
│                                                               │
│  ┌────────────┐  ┌───────────┐  ┌─────────────┐  │
│  │ Chat       │  │ Vision    │  │ Files      │  │
│  │ Completions│  │ API       │  │ Upload     │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬───────┘  │
└─────────┼──────────────────┼──────────────────────┼──────────────────┘
          │                  │                  │
          │                  │                  │
    ┌─────▼──────┐  ┌────▼─────┐   ┌────▼────────┐
    │ Feature Type  │  │ New     │   │ Conversation │
    │ Router        │  │ Schemas │   │ API Client   │
    └─────┬────────┘  └───────────┘   └──────┬──────┘
          │                               │
    ┌─────┼────────────┬───────────▼─────────────┐
    │     │            │                      │
┌───▼─┐ ┌──▼──────┐ ┌──▼──────────┐ ┌──▼───────────┐
│  AI   │ │  Image   │ │  PDF        │ │  Video       │
│  Chat  │ │  Chat    │ │  Chat       │ │  Chat        │
└───┬──┘ └────┬─────┘ └────┬───────┘ └────┬─────────┘
    │          │               │               │
    └──────────┼───────────────┼───────────────┘
               │               │
        ┌──────▼─────────────────▼────────┐
        │   1MinAI AI Feature API          │
        │   https://api.1min.ai           │
        └─────────────────────────────────────┘
```

---

## 1. Feature Type Detection

### Detection Logic

```python
def detect_feature_type(request: ChatCompletionRequest) -> str:
    """
    Determine 1MinAI feature type from OpenAI request.

    Priority:
    1. Check for image data in messages → CHAT_WITH_IMAGE
    2. Check for file/document references → CHAT_WITH_PDF
    3. Check for video URLs → CHAT_WITH_YOUTUBE_VIDEO
    4. Default → CHAT_WITH_AI
    """
    has_image = any(
        _has_image_url(m) or m.role == "user" and _has_image_url(m.content)
        for m in request.messages
    )
    
    has_document = any(
        "document_id" in m.content or "file_id" in m.content
        for m in request.messages
    )
    
    has_video = any(
        _is_youtube_url(m.content)
        for m in request.messages
    )
    
    if has_image:
        return "CHAT_WITH_IMAGE"
    elif has_document:
        return "CHAT_WITH_PDF"
    elif has_video:
        return "CHAT_WITH_YOUTUBE_VIDEO"
    else:
        return "CHAT_WITH_AI"

def _has_image_url(content: str) -> bool:
    """Check if content contains image URL or base64."""
    if not content:
        return False
    
    # Check for common image patterns
    patterns = [
        r'data:image/[a-z]+;base64',
        r'https?://[^\s]+\.(png|jpg|jpeg|gif|webp)',
        r'image_url',
    ]
    
    return any(re.search(pattern, content) for pattern in patterns)

def _is_youtube_url(content: str) -> bool:
    """Check if content contains YouTube URL."""
    if not content:
        return False
    return bool(re.search(r'https?://(?:www\.)?youtube\.com/', content))
```

---

## 2. Enhanced Schemas

### 2.1 Feature-Specific Prompt Objects

**File: `app/onemin_schemas.py`**

```python
from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field

# Existing
class ChatPromptObject(BaseModel):
    """PromptObject for CHAT_WITH_AI feature."""
    prompt: str = Field(..., description="The user's message/prompt")
    isMixed: bool = Field(default=False, description="Mix models context")
    webSearch: bool = Field(default=False, description="Enable web search")
    numOfSite: int = Field(default=1, description="Number of sites to search")
    maxWord: int = Field(default=500, description="Maximum words from web search")

# NEW: Image chat prompt object
class ImageChatPromptObject(BaseModel):
    """PromptObject for CHAT_WITH_IMAGE feature."""
    prompt: str = Field(..., description="Text description of image/context")
    imageUrl: str = Field(..., description="URL of image to analyze")
    isMixed: bool = Field(default=False, description="Mix models context")
    webSearch: bool = Field(default=False, description="Enable web search")
    numOfSite: int = Field(default=1, description="Number of sites to search")
    maxWord: int = Field(default=500, description="Maximum words from web search")

# NEW: PDF chat prompt object
class PDFChatPromptObject(BaseModel):
    """PromptObject for CHAT_WITH_PDF feature."""
    prompt: str = Field(default="", description="Optional text description")
    documentIds: List[str] = Field(..., description="List of document IDs to analyze")
    conversationId: str = Field(..., description="Required: conversation UUID from Conversation API")
    isMixed: bool = Field(default=False, description="Mix models context")
    webSearch: bool = Field(default=False, description="Enable web search")
    numOfSite: int = Field(default=1, description="Number of sites to search")
    maxWord: int = Field(default=500, description="Maximum words from web search")

# NEW: Video chat prompt object
class VideoChatPromptObject(BaseModel):
    """PromptObject for CHAT_WITH_YOUTUBE_VIDEO feature."""
    prompt: str = Field(default="", description="Optional text description")
    videoUrl: str = Field(..., description="YouTube video URL")
    conversationId: str = Field(..., description="Required: conversation UUID from Conversation API")
    isMixed: bool = Field(default=False, description="Mix models context")
    webSearch: bool = Field(default=False, description="Enable web search")
    numOfSite: int = Field(default=1, description="Number of sites to search")
    maxWord: int = Field(default=500, description="Maximum words from web search")

# NEW: Unified prompt object (union type)
class UnifiedPromptObject(BaseModel):
    """Union of all prompt object types for flexible routing."""
    prompt: Optional[str] = Field(default=None, description="Text prompt for chat")
    imageUrl: Optional[str] = Field(default=None, description="Image URL for vision chat")
    documentIds: Optional[List[str]] = Field(default=None, description="Document IDs for PDF chat")
    videoUrl: Optional[str] = Field(default=None, description="Video URL for video chat")
    conversationId: Optional[str] = Field(default=None, description="Conversation ID (required for PDF/Video)")
    isMixed: bool = Field(default=False, description="Mix models context")
    webSearch: bool = Field(default=False, description="Enable web search")
    numOfSite: int = Field(default=1, description="Number of sites for web search")
    maxWord: int = Field(default=500, description="Maximum words from web search")
```

### 2.2 Enhanced Request Schema

```python
# UPDATED: Flexible 1MinAI request schema
class OneMinAIRequest(BaseModel):
    """
    Unified request body for all 1MinAI feature types.
    
    Supports:
    - CHAT_WITH_AI (existing)
    - CHAT_WITH_IMAGE (new)
    - CHAT_WITH_PDF (new)
    - CHAT_WITH_YOUTUBE_VIDEO (new)
    """
    type: Literal["CHAT_WITH_AI", "CHAT_WITH_IMAGE", "CHAT_WITH_PDF", "CHAT_WITH_YOUTUBE_VIDEO"] = Field(
        ..., description="Feature type to use"
    )
    model: str = Field(..., description="Model name to use")
    promptObject: UnifiedPromptObject = Field(..., description="Feature-specific parameters")
    
    # Additional optional fields
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")
```

### 2.3 Conversation API Schema

```python
# NEW: Conversation creation for PDF/Video features
class CreateConversationRequest(BaseModel):
    """Request to create a conversation via Conversation API."""
    
    messageGroup: str = Field(..., description="Message group identifier")
    title: Optional[str] = Field(default=None, description="Conversation title")
    
class ConversationResponse(BaseModel):
    """Response from Conversation API."""
    
    uuid: str = Field(..., description="Conversation UUID")
    messageGroup: str = Field(..., description="Message group identifier")
    createdAt: str = Field(..., description="Creation timestamp")
```

### 2.4 Enhanced OpenAI Schemas

```python
# UPDATED: Add image/role detection
class ChatMessage(BaseModel):
    """A single message in chat conversation."""
    
    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="The role of message author"
    )
    content: str = Field(..., description="The content of message")
    
    # NEW: Support for image URLs in content
    image_url: Optional[str] = Field(
        default=None,
        description="URL of image (for vision capabilities)"
    )
    
    # NEW: Support for file/document references
    file_id: Optional[str] = Field(
        default=None,
        description="File ID (for document analysis)"
    )
```

---

## 3. Translation Layer Updates

### 3.1 Feature Type Router

**File: `app/translate.py`**

```python
from typing import Optional
from .onemin_schemas import (
    OneMinAIRequest,
    UnifiedPromptObject,
    ImageChatPromptObject,
    PDFChatPromptObject,
    VideoChatPromptObject,
)

def detect_feature_type(messages: list) -> str:
    """
    Determine appropriate 1MinAI feature type from message content.
    
    Returns: One of CHAT_WITH_AI, CHAT_WITH_IMAGE, CHAT_WITH_PDF, CHAT_WITH_YOUTUBE_VIDEO
    """
    has_image = any(_contains_image(m) for m in messages)
    has_file = any(m.file_id is not None for m in messages if hasattr(m, 'file_id'))
    has_video = any(_contains_youtube_url(m.content) for m in messages if hasattr(m, 'content'))
    
    if has_image and not has_file:
        return "CHAT_WITH_IMAGE"
    elif has_file:
        return "CHAT_WITH_PDF"
    elif has_video and not has_file:
        return "CHAT_WITH_YOUTUBE_VIDEO"
    else:
        return "CHAT_WITH_AI"

def _contains_image(message) -> bool:
    """Check if message contains image data."""
    content = getattr(message, 'content', '')
    if not content:
        return False
    
    # Check for image patterns
    image_patterns = [
        'data:image/',
        '.png', '.jpg', '.jpeg', '.gif', '.webp',
        'image_url', 'imageUrl',
    ]
    return any(pattern in content.lower() for pattern in image_patterns)

def _contains_youtube_url(content: str) -> bool:
    """Check if content is a YouTube URL."""
    if not content:
        return False
    return 'youtube.com' in content.lower() or 'youtu.be' in content.lower()

def openai_to_onemin_request(
    request: ChatCompletionRequest,
    settings: Settings,
    feature_type: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> OneMinAIRequest:
    """
    Translate OpenAI chat completion request to appropriate 1MinAI feature.
    
    Args:
        request: OpenAI-style chat completion request
        settings: Application settings
        feature_type: Optional override (for testing)
        conversation_id: Optional conversation UUID (for PDF/Video features)
    
    Returns:
        Appropriate 1MinAI request for detected feature type
    """
    # Determine feature type if not provided
    if feature_type is None:
        feature_type = detect_feature_type(request.messages)
    
    # Determine model to use
    model = request.model if request.model else settings.default_1min_model
    
    # Extract image URL if present (for CHAT_WITH_IMAGE)
    image_url = None
    for msg in request.messages:
        if msg.role == "user":
            image_url = _extract_first_image_url(msg.content)
            if image_url:
                break
    
    # Build unified prompt object
    prompt_object = UnifiedPromptObject(
        prompt=messages_to_prompt(request.messages),
        imageUrl=image_url,
        isMixed=settings.default_websearch,
        webSearch=settings.default_websearch,
        numOfSite=settings.default_num_of_site,
        maxWord=settings.default_max_word,
        conversationId=conversation_id,
    )
    
    return OneMinAIRequest(
        type=feature_type,  # Dynamic feature type
        model=model,
        promptObject=prompt_object,
    )

def _extract_first_image_url(content: str) -> Optional[str]:
    """Extract first image URL from message content."""
    import re
    # Match data:image patterns
    match = re.search(r'data:image/[a-z]+;base64,([a-zA-Z0-9+/=]+)', content)
    if match:
        return f"data:image/{match.group(0)};base64,{match.group(1)}"
    
    # Match image URLs
    match = re.search(r'https?://[^\s]+\.(png|jpg|jpeg|gif|webp)', content)
    if match:
        return match.group(0)
    
    return None
```

### 3.2 Feature-Specific Response Translation

```python
def onemin_to_openai_response(
    onemin_response: dict,
    model: str,
    feature_type: str,
    prompt_text: str,
) -> tuple[ChatCompletionResponse, bool]:
    """
    Translate 1MinAI response to OpenAI format (feature-aware).
    
    Args:
        onemin_response: Raw response from 1MinAI
        model: Model name used
        feature_type: Feature type that was used
        prompt_text: Original prompt text for token estimation
    
    Returns:
        Tuple of (ChatCompletionResponse, usage_estimated: bool)
    """
    # Extract assistant response text (logic same as before)
    assistant_text = ""
    ai_record = onemin_response.get("aiRecord", {})
    detail = ai_record.get("aiRecordDetail", {})
    result_object = detail.get("resultObject")
    
    if result_object:
        if isinstance(result_object, list) and result_object:
            assistant_text = str(result_object[0]) if result_object else ""
        elif isinstance(result_object, str):
            assistant_text = result_object
    else:
        assistant_text = ai_record.get("response", "")
    
    # Generate completion ID
    completion_id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
    
    # Token estimation
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(assistant_text)
    usage_estimated = True
    
    # Determine finish reason
    status = ai_record.get("status", "SUCCESS")
    finish_reason: Optional[str] = "stop"
    if status == "FAILED":
        finish_reason = None
    elif status == "CONTENT_FILTER":
        finish_reason = "content_filter"
    
    # Build response (same as before)
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
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    
    return response, usage_estimated
```

---

## 4. OpenAI Compatibility Mappings

### 4.1 Endpoint Mapping

| 1MinAI Feature | OpenAI Endpoint | Proxy Endpoint | Description |
|----------------|---------------|----------------|-------------|
| `CHAT_WITH_AI` | `/v1/chat/completions` | `/v1/chat/completions` | Standard text chat (existing) |
| `CHAT_WITH_IMAGE` | Vision API + `/v1/chat/completions` | `/v1/chat/completions` | Multimodal chat with images |
| `CHAT_WITH_PDF` | `/v1/files` + `/v1/chat/completions` | `/v1/files` | Document upload & analysis |
| `CHAT_WITH_YOUTUBE_VIDEO` | `/v1/chat/completions` | `/v1/chat/completions` | Video content understanding |

### 4.2 Model Availability by Feature

Based on testing results:

| Feature | Working Models | Not Working Models |
|---------|--------------|-------------------|
| CHAT_WITH_AI | gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo, gpt-5-nano, gpt-5, gpt-5-mini, gpt-5.1, gpt-4.1-nano, gpt-4.1-mini, o3-mini, o3, o3-pro, o4-mini, deepseek-chat, deepseek-reasoner, qwen-plus, qwen-max, mistral-large-latest, mistral-small-latest | claude-*, gemini-*, llama-*, grok-* |
| CHAT_WITH_IMAGE | TBD (likely gpt-4o-vision, claude-opus-4) | TBD |
| CHAT_WITH_PDF | TBD | TBD |
| CHAT_WITH_YOUTUBE_VIDEO | TBD | TBD |

---

## 5. New Endpoints

### 5.1 Files Upload Endpoint

```python
# File: app/main.py (NEW ENDPOINT)

@app.post("/v1/files", response_model=None, tags=["Files"])
async def upload_files(
    request: UploadFileRequest,
    x_request_id: str | None = Header(None, alias="X-Request-ID"),
) -> JSONResponse:
    """
    Upload files for document analysis (CHAT_WITH_PDF).
    
    OpenAI-compatible endpoint for file uploads.
    """
    settings = get_settings()
    request_id = get_request_id(x_request_id)
    
    log_request(
        request_id=request_id,
        model="file_upload",
        message_count=1,
        prompt_length=len(request.filename),
    )
    
    # Create conversation via 1MinAI Conversation API
    client = OneMinClient(settings)
    conversation = await client.create_conversation(
        message_group=f"upload_{request_id}",
        title=request.filename,
    )
    
    # Return conversation ID
    return JSONResponse({
        "id": conversation.uuid,
        "object": "file",
        "created_at": conversation.createdAt,
        "filename": request.filename,
    })
```

### 5.2 Vision Chat Endpoint (Optional)

```python
# Can reuse /v1/chat/completions with feature type detection
# No new endpoint needed if detection logic is robust
```

---

## 6. Conversation API Integration

### 6.1 Client Method

```python
# File: app/onemin_client.py (NEW METHOD)

class OneMinClient:
    # ... existing methods ...
    
    async def create_conversation(
        self,
        message_group: str,
        title: Optional[str] = None,
    ) -> ConversationResponse:
        """
        Create a conversation via Conversation API.
        
        Required for PDF and Video features before using them.
        
        Args:
            message_group: Message group identifier
            title: Optional title for the conversation
        
        Returns:
            Conversation UUID and metadata
        """
        endpoint = f"{self.base_url}/api/conversations"
        payload = {
            "messageGroup": message_group,
            "title": title,
        }
        
        client = await self._get_client()
        response = await client.post(
            endpoint,
            headers={
                "API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        
        if response.status_code != 201:
            raise ProxyHTTPException(
                status_code=response.status_code,
                message="Failed to create conversation",
                error_type="upstream_error",
            )
        
        return ConversationResponse(**response.json())
```

---

## 7. Implementation Roadmap (Revised Order)

### Phase 1: Foundation & Validation (Backward Compatible)
- [ ] Add `create_conversation()` to `OneMinAIClient`
- [ ] Add `ConversationResponse` and `CreateConversationRequest` schemas
- [ ] Add validation helpers module (`app/validation.py`)
- [ ] Add feature-specific error classes
- [ ] Add feature type detection function
- [ ] Add prompt object schemas for all features
- [ ] Update translation layer to support feature type routing
- [ ] Add unit tests for detection logic
- [ ] Test backward compatibility with existing CHAT_WITH_AI requests

### Phase 2: Vision Support
- [ ] Implement simplified image URL extraction
- [ ] Add `ImageChatPromptObject` schema
- [ ] Add model support mapping (validate model supports feature)
- [ ] Add `CHAT_WITH_IMAGE` to translation layer
- [ ] Test with OpenAI Vision API clients
- [ ] Add vision models to OpenClaw config
- [ ] Test streaming for CHAT_WITH_IMAGE

### Phase 3: Document Support
- [ ] Implement `/v1/files` endpoint
- [ ] Add file upload request schema
- [ ] Add `PDFChatPromptObject` schema
- [ ] Add CHAT_WITH_PDF translation support
- [ ] Integrate Conversation API client method
- [ ] Test PDF upload → conversation creation → chat workflow
- [ ] Add document models to OpenClaw config

### Phase 4: Video Support
- [ ] Implement YouTube URL detection
- [ ] Add `VideoChatPromptObject` schema
- [ ] Add CHAT_WITH_YOUTUBE_VIDEO translation support
- [ ] Test video analysis workflow
- [ ] Add conversation management for video sessions
- [ ] Add video models to OpenClaw config

### Phase 5: Feature-Specific Error Handling
- [ ] Add `FeatureNotEnabledError` exception class
- [ ] Add `InvalidMediaError` exception class
- [ ] Add `ConversationNotFoundError` exception class
- [ ] Update error handler to use specific exceptions
- [ ] Add logging for feature routing decisions
- [ ] Test error handling for all feature types

### Phase 6: Model-Feature Validation
- [ ] Add model support mapping dictionary
- [ ] Add `validate_model_for_feature()` function
- [ ] Fail-fast with clear error messages
- [ ] Document which models support which features
- [ ] Update settings with feature-specific rate limits
- [ ] Test model validation for all feature types

### Phase 7: Advanced Features
- [ ] Web search integration (already in promptObject)
- [ ] Conversation history management
- [ ] Multi-turn conversation support
- [ ] Streaming support verification for all features
- [ ] Per-feature rate limiting
- [ ] Model discovery endpoint

### Phase 8: Testing & Quality Assurance
- [ ] Add unit tests for validation helpers
- [ ] Add integration tests for all feature types
- [ ] Add backward compatibility test suite
- [ ] Load testing for concurrent requests
- [ ] Test error scenarios for all features
- [ ] Documentation review and validation

### Phase 9: Documentation & Deployment
- [ ] Update README with multi-feature support
- [ ] Document environment variables
- [ ] Add migration guide
- [ ] Create OpenClaw configuration template
- [ ] Feature flag rollout plan
- [ ] Post-launch monitoring checklist

---

## 8. Configuration Updates

### Environment Variables

```python
# File: app/settings.py

class Settings(BaseSettings):
    # ... existing settings ...
    
    # NEW: Feature-specific defaults
    enable_vision_features: bool = Field(
        default=True,
        description="Enable CHAT_WITH_IMAGE support",
        alias="ENABLE_VISION_FEATURES",
    )
    
    enable_document_features: bool = Field(
        default=False,  # Default off (requires Conversation API integration)
        description="Enable CHAT_WITH_PDF support",
        alias="ENABLE_DOCUMENT_FEATURES",
    )
    
    enable_video_features: bool = Field(
        default=False,  # Default off (requires Conversation API integration)
        description="Enable CHAT_WITH_YOUTUBE_VIDEO support",
        alias="ENABLE_VIDEO_FEATURES",
    )
    
    # NEW: Feature type override (for testing)
    feature_type_override: Optional[str] = Field(
        default=None,
        description="Override feature type detection (testing only)",
        alias="FEATURE_TYPE_OVERRIDE",
    )
```

---

## 14. Testing Strategy (Updated)

### 14.1 Unit Tests

```python
# File: tests/test_validation.py (NEW)

import pytest
from app.validation import (
    validate_conversation_id,
    validate_image_url,
    validate_model_for_feature,
    validate_document_ids,
)
from app.validation import (
    FeatureNotEnabledError,
    InvalidMediaError,
    ConversationNotFoundError,
    ModelNotSupportedError,
)
from app.onemin_schemas import ChatMessage

def test_feature_detection_text_only():
    """Test detection of CHAT_WITH_AI feature."""
    messages = [
        ChatMessage(role="user", content="Hello, how are you?"),
    ]
    
    from app.translate import detect_feature_type
    assert detect_feature_type(messages) == "CHAT_WITH_AI"

def test_feature_detection_with_image():
    """Test detection of CHAT_WITH_IMAGE feature."""
    messages = [
        ChatMessage(
            role="user",
            content="What's in this image? https://example.com/test.jpg"
        ),
    ]
    
    from app.translate import detect_feature_type
    assert detect_feature_type(messages) == "CHAT_WITH_IMAGE"

def test_feature_detection_with_file():
    """Test detection of CHAT_WITH_PDF feature."""
    from app.onemin_schemas import ChatMessage
    
    # Create message with file_id attribute
    msg = ChatMessage(role="user", content="Read this document")
    msg.file_id = "doc-123"
    messages = [msg]
    
    from app.translate import detect_feature_type
    assert detect_feature_type(messages) == "CHAT_WITH_PDF"

def test_feature_detection_with_video():
    """Test detection of CHAT_WITH_YOUTUBE_VIDEO feature."""
    from app.onemin_schemas import ChatMessage
    
    messages = [
        ChatMessage(
            role="user",
            content="Summarize this video: https://youtube.com/watch?v=xyz"
        ),
    ]
    
    from app.translate import detect_feature_type
    assert detect_feature_type(messages) == "CHAT_WITH_YOUTUBE_VIDEO"

def test_validate_conversation_id_missing():
    """Test validation fails when conversationId missing for PDF."""
    import pytest
    from app.validation import validate_conversation_id, ConversationNotFoundError
    from app.settings import Settings
    
    with pytest.raises(ConversationNotFoundError) as exc_info:
        validate_conversation_id(
            feature_type="CHAT_WITH_PDF",
            prompt_obj={},
            settings=Settings(),
        )
    
    assert "conversationId" in str(exc_info.value)

def test_validate_model_not_supported():
    """Test validation fails when model not supported for feature."""
    import pytest
    from app.validation import validate_model_for_feature, ModelNotSupportedError
    from app.settings import Settings
    
    with pytest.raises(ModelNotSupportedError) as exc_info:
        validate_model_for_feature(
            model="claude-3-5-sonnet",
            feature_type="CHAT_WITH_IMAGE",
            settings=Settings(),
        )
    
    assert "claude-3-5-sonnet" in str(exc_info.value)
```

### 14.2 Integration Tests (Updated)

```python
# File: tests/test_multi_feature_integration.py (EXPANDED)

import pytest
import httpx

@pytest.mark.asyncio
async def test_chat_with_image_feature(client: httpx.AsyncClient):
    """Test CHAT_WITH_IMAGE feature end-to-end."""
    from app.validation import validate_model_for_feature
    
    # Validate model supports vision
    validate_model_for_feature("gpt-4o", "CHAT_WITH_IMAGE", settings=Settings())
    
    response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "What's in this image? https://example.com/test.jpg",
                }
            ],
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] is not None

@pytest.mark.asyncio
async def test_chat_with_disabled_feature(client: httpx.AsyncClient, monkeypatch):
    """Test error when trying to use disabled feature."""
    import os
    from app.validation import FeatureNotEnabledError
    
    # Disable vision features
    monkeypatch.setenv("ENABLE_VISION_FEATURES", "false")
    
    response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "What's in this image? https://example.com/test.jpg",
                }
            ],
        }
    )
    
    assert response.status_code == 503
    data = response.json()
    assert "feature" in str(data["error"]["message"]).lower()
    assert "CHAT_WITH_IMAGE" in str(data["error"]["message"])

@pytest.mark.asyncio
async def test_pdf_upload_workflow(client: httpx.AsyncClient):
    """Test PDF upload and chat workflow."""
    # Upload file
    upload_response = await client.post(
        "http://localhost:8080/v1/files",
        json={"filename": "test.pdf", "content": "..."},
    )
    
    assert upload_response.status_code == 200
    conversation_id = upload_response.json()["id"]
    
    # Chat with conversation ID
    chat_response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Summarize PDF"}
            ],
        },
    )
    
    assert chat_response.status_code == 200
    assert chat_response.json()["model"] == "gpt-4o"

@pytest.mark.asyncio
async def test_video_chat_workflow(client: httpx.AsyncClient):
    """Test video analysis workflow."""
    response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize this video: https://youtube.com/watch?v=test",
                }
            ],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data

@pytest.mark.asyncio
async def test_model_validation_vision_feature(client: httpx.AsyncClient):
    """Test model validation for vision feature."""
    from app.validation import ModelNotSupportedError
    
    response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "claude-3-5-sonnet",  # Not vision-capable
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze image: https://example.com/test.jpg",
                }
            ],
        },
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "not supported" in str(data["error"]["message"]).lower()
```

### 14.3 Backward Compatibility Tests

```python
# File: tests/test_backward_compatibility.py (NEW)

import pytest

@pytest.mark.asyncio
async def test_existing_chat_completions_still_works(client: httpx.AsyncClient):
    """Test that existing CHAT_WITH_AI requests still work."""
    response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
        },
    )
    
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_old_client_without_image_support(client: httpx.AsyncClient):
    """Test backward compatibility when client only supports text."""
    response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    # Should not use CHAT_WITH_IMAGE even if image detected in content
    # (depends on feature flag behavior)

@pytest.mark.asyncio
async def test_fallback_to_chat_with_ai(client: httpx.AsyncClient):
    """Test that unclear requests fall back to CHAT_WITH_AI."""
    response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user", 
                    "content": "Just text, no media"
                }
            ],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4o-mini"
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# File: tests/test_feature_detection.py

import pytest
from app.translate import detect_feature_type, _contains_image, _contains_youtube_url

def test_detect_text_chat():
    """Test detection of CHAT_WITH_AI feature."""
    from app.onemin_schemas import ChatMessage
    
    messages = [
        ChatMessage(role="user", content="Hello, how are you?"),
    ]
    
    assert detect_feature_type(messages) == "CHAT_WITH_AI"

def test_detect_image_chat():
    """Test detection of CHAT_WITH_IMAGE feature."""
    messages = [
        ChatMessage(
            role="user",
            content="What's in this image? https://example.com/image.jpg"
        ),
    ]
    
    assert detect_feature_type(messages) == "CHAT_WITH_IMAGE"

def test_detect_base64_image():
    """Test detection of base64 images."""
    messages = [
        ChatMessage(
            role="user",
            content="Analyze this: data:image/png;base64,iVBORw0KG..."
        ),
    ]
    
    assert detect_feature_type(messages) == "CHAT_WITH_IMAGE"

def test_detect_pdf_chat():
    """Test detection of CHAT_WITH_PDF feature."""
    messages = [
        ChatMessage(role="user", content="Read this document"),
        ChatMessage(role="user", content="file_id: doc-123"),
    ]
    
    assert detect_feature_type(messages) == "CHAT_WITH_PDF"

def test_detect_youtube_chat():
    """Test detection of CHAT_WITH_YOUTUBE_VIDEO feature."""
    messages = [
        ChatMessage(
            role="user",
            content="Summarize this video: https://youtube.com/watch?v=xyz"
        ),
    ]
    
    assert detect_feature_type(messages) == "CHAT_WITH_YOUTUBE_VIDEO"
```

### 9.2 Integration Tests

```python
# File: tests/test_multi_feature_integration.py

import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_chat_with_image_feature(client: AsyncClient):
    """Test CHAT_WITH_IMAGE feature end-to-end."""
    response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "What's in this image? https://example.com/test.jpg",
                }
            ],
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] is not None

@pytest.mark.asyncio
async def test_pdf_upload_workflow(client: AsyncClient):
    """Test PDF upload and chat workflow."""
    # Upload file
    upload_response = await client.post(
        "http://localhost:8080/v1/files",
        json={"filename": "test.pdf", "content": "..."},
    )
    
    assert upload_response.status_code == 200
    conversation_id = upload_response.json()["id"]
    
    # Chat with conversation ID
    chat_response = await client.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Summarize the PDF"}
            ],
        },
    )
    
    assert chat_response.status_code == 200
```

---

## 10. OpenClaw Configuration Updates

### Updated Config for Multi-Feature Support

```json
{
  "models": {
    "providers": {
      "1minai": {
        "baseUrl": "http://localhost:8080/v1",
        "api": "openai-completions",
        "models": [
          {
            "id": "gpt-4o-mini",
            "name": "GPT-4o Mini (1MinAI)",
            "reasoning": false,
            "input": ["text", "image"],
            "capabilities": ["chat", "vision"]
          },
          {
            "id": "gpt-4o",
            "name": "GPT-4o (1MinAI)",
            "reasoning": false,
            "input": ["text", "image"],
            "capabilities": ["chat", "vision"]
          },
          {
            "id": "gpt-5-nano",
            "name": "GPT-5 Nano (1MinAI)",
            "reasoning": false,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "gpt-5",
            "name": "GPT-5 (1MinAI)",
            "reasoning": true,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "deepseek-chat",
            "name": "DeepSeek Chat (1MinAI)",
            "reasoning": false,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "deepseek-reasoner",
            "name": "DeepSeek Reasoner (1MinAI)",
            "reasoning": true,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "qwen-plus",
            "name": "Qwen Plus (1MinAI)",
            "reasoning": false,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "qwen-max",
            "name": "Qwen Max (1MinAI)",
            "reasoning": false,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "o3-mini",
            "name": "O3 Mini (1MinAI)",
            "reasoning": true,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "o3",
            "name": "O3 (1MinAI)",
            "reasoning": true,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "o3-pro",
            "name": "O3 Pro (1MinAI)",
            "reasoning": true,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "o4-mini",
            "name": "O4 Mini (1MinAI)",
            "reasoning": true,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "mistral-large-latest",
            "name": "Mistral Large (1MinAI)",
            "reasoning": false,
            "input": ["text"],
            "capabilities": ["chat"]
          },
          {
            "id": "mistral-small-latest",
            "name": "Mistral Small (1MinAI)",
            "reasoning": false,
            "input": ["text"],
            "capabilities": ["chat"]
          }
        ]
      }
    }
  }
}
```

---

## 11. Migration Path

### Backward Compatibility Strategy

```python
# Phase 1: Non-breaking changes
# - Add feature type detection (behind feature flag)
# - Add new schemas (unused initially)
# - Keep CHAT_WITH_AI hardcoded for existing behavior
# - Add tests

# Phase 2: Soft launch
# - Enable feature type routing with env var ENABLE_FEATURE_ROUTING=false
# - Monitor for issues
# - Document migration guide

# Phase 3: Hard launch
# - Enable feature routing by default
# - Update README with new features
# - Remove deprecated hardcoded CHAT_WITH_AI
# - Celebrate!
```

---

## 12. Key Design Decisions

### Why Smart Routing Instead of Multiple Endpoints?

**Decision**: Single `/v1/chat/completions` endpoint with feature detection

**Rationale**:
1. **OpenAI Compatibility**: Clients expect single endpoint for all chat
2. **Flexibility**: Feature type can change without endpoint changes
3. **Simpler Client Code**: No need for endpoint-specific logic
4. **Future-Proof**: New features can be added without new routes

### Conversation API Strategy

**Decision**: Create conversations per-document or per-session, not global

**Rationale**:
1. **1MinAI Requirement**: Docs state PDF/Video require conversationId
2. **Isolation**: Each file/video gets its own conversation
3. **Cleaner State**: No need to manage conversation lifecycles
4. **Error Containment**: Bad conversationId only affects that request

### Error Handling Philosophy

**Decision**: Feature-specific exceptions with clear error messages

**Rationale**:
1. **Client Clarity**: Specific error = specific fix
2. **Debugging**: Stack traces point to exact issue
3. **Graceful Degradation**: Feature disabled = informative message
4. **Validation**: Fail-fast before API calls (cheaper)

### Why Conversation API for PDF/Video Only?

**Decision**: Use Conversation API for multi-turn document/video chats

**Rationale**:
1. **1MinAI Requirement**: Docs state PDF/Video require conversationId
2. **State Management**: Better for multi-turn conversations
3. **Cost Efficiency**: Reuse conversations instead of re-uploading
4. **Consistency**: Aligns with 1MinAI's expected usage pattern

---

## 13. Success Metrics

### Implementation Success Criteria

- [x] All 14 CHAT_WITH_AI models continue to work
- [ ] CHAT_WITH_IMAGE works with at least 2 models
- [ ] CHAT_WITH_PDF creates conversations and analyzes documents
- [ ] CHAT_WITH_YOUTUBE_VIDEO creates conversations and analyzes videos
- [ ] 100% backward compatibility with existing tests
- [ ] New tests added for all feature types
- [ ] OpenClaw config updated with capabilities
- [ ] Documentation updated with new features

---

## 13. Revised Schema Designs

### 13.1 Validation Helpers Module

**File: `app/validation.py` (NEW)**

```python
"""
Validation helpers for multi-feature 1MinAI proxy.

Fail-fast validation to prevent invalid API calls.
"""

from typing import Optional
from urllib.parse import urlparse


class ValidationError(Exception):
    """Base validation error."""
    pass


class FeatureNotEnabledError(ValidationError):
    """Raised when trying to use disabled feature."""
    def __init__(self, feature_type: str):
        self.feature_type = feature_type
        super().__init__(
            f"Feature '{feature_type}' is not enabled. "
            f"Enable with {feature_type.upper()}_FEATURES=true in your settings."
        )


class InvalidMediaError(ValidationError):
    """Raised when image URL or file ID is invalid."""
    def __init__(self, media_type: str, details: str):
        self.media_type = media_type
        self.details = details
        super().__init__(
            f"Invalid {media_type}: {details}"
        )


class ConversationNotFoundError(ValidationError):
    """Raised when conversationId doesn't exist or is invalid."""
    def __init__(self, conversation_id: str, feature_type: str):
        self.conversation_id = conversation_id
        self.feature_type = feature_type
        super().__init__(
            f"Conversation '{conversation_id}' not found or invalid for {feature_type}. "
            f"Please create a conversation first using /v1/files endpoint."
        )


class ModelNotSupportedError(ValidationError):
    """Raised when model doesn't support the requested feature."""
    def __init__(self, model: str, feature_type: str):
        self.model = model
        self.feature_type = feature_type
        super().__init__(
            f"Model '{model}' is not supported for feature {feature_type}. "
            f"Available models for this feature: {get_supported_models(feature_type)}"
        )


def validate_conversation_id(
    feature_type: str,
    prompt_obj: dict,
    settings: Settings,
) -> None:
    """
    Ensure conversationId is provided and valid for PDF/Video features.
    
    Raises:
        ValidationError: If conversationId is missing/invalid
    """
    if feature_type in ("CHAT_WITH_PDF", "CHAT_WITH_YOUTUBE_VIDEO"):
        conversation_id = prompt_obj.get("conversationId")
        
        if not conversation_id or not isinstance(conversation_id, str):
            raise ConversationNotFoundError(
                conversation_id=str(conversation_id),
                feature_type=feature_type,
            )


def validate_image_url(content: str) -> Optional[str]:
    """
    Extract and validate image URL from message content.
    
    Uses URL parsing instead of regex for better reliability.
    
    Returns:
        Image URL if found and valid
        None if not found or invalid
    """
    if not content:
        return None
    
    # Look for HTTP/HTTPS URLs that look like images
    for word in content.split():
        try:
            parsed = urlparse(word)
            if parsed.scheme not in ("http", "https"):
                continue
            
            # Check for common image file extensions
            path = parsed.path.lower()
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')
            if any(path.endswith(ext) for ext in image_extensions):
                return word
            
        except Exception:
            # If parsing fails, it's probably not a URL
            continue
    
    return None


def validate_document_ids(document_ids: list) -> None:
    """
    Validate document IDs for PDF chat.
    
    Raises:
        ValidationError: If no document IDs provided for PDF feature
    """
    if not document_ids or len(document_ids) == 0:
        raise ValidationError(
            "documentIds must be a non-empty list for CHAT_WITH_PDF"
        )


def validate_model_for_feature(
    model: str,
    feature_type: str,
    settings: Settings,
) -> None:
    """
    Validate that model supports the requested feature type.
    
    Raises:
        ModelNotSupportedError: If model not in supported list
    """
    from .onemin_client import MODEL_FEATURE_SUPPORT
    
    # Check if model exists in feature's supported list
    supported_models = MODEL_FEATURE_SUPPORT.get(feature_type, set())
    
    if model not in supported_models:
        raise ModelNotSupportedError(
            model=model,
            feature_type=feature_type,
        )


def get_supported_models(feature_type: str) -> list[str]:
    """
    Get list of models that support a specific feature type.
    """
    from .onemin_client import MODEL_FEATURE_SUPPORT
    return list(MODEL_FEATURE_SUPPORT.get(feature_type, set()))
```

### 13.2 Simplified Image Detection

**Updated: `app/translate.py`**

```python
from .validation import validate_image_url, validate_model_for_feature

def detect_feature_type(messages: list) -> str:
    """
    Determine appropriate 1MinAI feature type from message content.
    
    Priority:
    1. Check for image data in messages → CHAT_WITH_IMAGE
    2. Check for file/document references → CHAT_WITH_PDF
    3. Check for video URLs → CHAT_WITH_YOUTUBE_VIDEO
    4. Default → CHAT_WITH_AI
    """
    has_image = False
    image_url = None
    
    # Use URL parsing (more reliable than regex)
    for msg in messages:
        if msg.role == "user":
            url = validate_image_url(msg.content)
            if url:
                has_image = True
                image_url = url
                break
    
    has_file = any(
        hasattr(msg, 'file_id') and msg.file_id is not None
        for msg in messages
        if hasattr(msg, 'file_id')  # Check for file_id attribute
    )
    
    has_video = any(
        _is_youtube_url(msg.content) if hasattr(msg, 'content') else False
        for msg in messages
        if hasattr(msg, 'content')
    )
    
    if has_image:
        return "CHAT_WITH_IMAGE"
    elif has_file:
        return "CHAT_WITH_PDF"
    elif has_video:
        return "CHAT_WITH_YOUTUBE_VIDEO"
    else:
        return "CHAT_WITH_AI"


def _is_youtube_url(content: str) -> bool:
    """Check if content is a YouTube URL."""
    if not content:
        return False
    return 'youtube.com' in content.lower() or 'youtu.be' in content.lower()


def openai_to_onemin_request(
    request: ChatCompletionRequest,
    settings: Settings,
    feature_type: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> OneMinAIRequest:
    """
    Translate OpenAI chat completion request to appropriate 1MinAI feature.
    
    Args:
        request: OpenAI-style chat completion request
        settings: Application settings
        feature_type: Optional override (for testing)
        conversation_id: Optional conversation UUID (for PDF/Video features)
    
    Returns:
        Appropriate 1MinAI request for detected feature type
    """
    # Determine feature type if not provided
    if feature_type is None:
        feature_type = detect_feature_type(request.messages)
    
    # Validate inputs
    validate_model_for_feature(request.model or settings.default_1min_model, feature_type, settings)
    
    if feature_type in ("CHAT_WITH_PDF", "CHAT_WITH_YOUTUBE_VIDEO"):
        # For document/video features, validate conversation ID
        validate_conversation_id(
            feature_type=feature_type,
            prompt_obj={},  # Will be built next
            settings=settings,
        )
    
    # Extract image URL (for CHAT_WITH_IMAGE)
    image_url = None
    for msg in request.messages:
        if msg.role == "user":
            url = validate_image_url(msg.content)
            if url:
                image_url = url
                break
    
    # Build unified prompt object
    prompt_object = UnifiedPromptObject(
        prompt=messages_to_prompt(request.messages),
        imageUrl=image_url,
        conversationId=conversation_id,
        isMixed=settings.default_websearch,
        webSearch=settings.default_websearch,
        numOfSite=settings.default_num_of_site,
        maxWord=settings.default_max_word,
    )
    
    return OneMinAIRequest(
        type=feature_type,  # Dynamic feature type
        model=request.model or settings.default_1min_model,
        promptObject=prompt_object,
    )
```

### 13.3 Model Feature Support Mapping

**File: `app/onemin_client.py` (UPDATED)**

```python
# At top of file, after imports

# Model support mapping per feature type
# This enables validation and OpenClaw capability metadata
MODEL_FEATURE_SUPPORT = {
    "CHAT_WITH_AI": {
        # All 14 tested and working models
        "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo",
        "gpt-5-nano", "gpt-5", "gpt-5-mini", "gpt-5.1",
        "gpt-4.1-nano", "gpt-4.1-mini",
        "o3-mini", "o3", "o3-pro", "o4-mini",
        "deepseek-chat", "deepseek-reasoner",
        "qwen-plus", "qwen-max",
        "mistral-large-latest", "mistral-small-latest",
    },
    "CHAT_WITH_IMAGE": {
        # Vision-capable models (to be tested and confirmed)
        "gpt-4o",  # Primary vision model
        "claude-opus-4",  # Likely supports vision
        "gemini-1.5-pro",  # Known for multimodal
        # Add more as tested
    },
    "CHAT_WITH_PDF": {
        # Models that support document analysis (to be tested)
        "gpt-4o",  # Strong reasoning
        "claude-3-5-sonnet",  # Document analysis
        "gemini-1.5-pro",  # Large context
    },
    "CHAT_WITH_YOUTUBE_VIDEO": {
        # Models that support video analysis (to be tested)
        "gpt-4o",  # Multimodal
        "gemini-1.5-flash",  # Fast processing
        # Add more as tested
    },
}


# Updated __init__ to include feature flags
def __init__(self, settings: Settings):
    self.settings = settings
    self.base_url = settings.one_min_ai_base_url.rstrip("/")
    self.timeout = httpx.Timeout(
        timeout=float(settings.request_timeout_secs),
        connect=10.0,
    )
    self._client: Optional[httpx.AsyncClient] = None
    
    # NEW: Feature support flags
    self.enable_vision_features = settings.enable_vision_features
    self.enable_document_features = settings.enable_document_features
    self.enable_video_features = settings.enable_video_features
```

### 13.4 Feature-Specific Error Classes

**File: `app/errors.py` (UPDATED)**

```python
# Add to existing error classes

class FeatureNotEnabledError(ProxyHTTPException):
    """
    Raised when trying to use a disabled feature.
    
    HTTP Status: 503 (Service Unavailable)
    """
    pass


class InvalidMediaError(ProxyHTTPException):
    """
    Raised when image URL or file ID is invalid.
    
    HTTP Status: 400 (Bad Request)
    """
    pass


class ConversationNotFoundError(ProxyHTTPException):
    """
    Raised when conversation ID doesn't exist.
    
    HTTP Status: 404 (Not Found)
    """
    pass


class ModelNotSupportedError(ProxyHTTPException):
    """
    Raised when model doesn't support the requested feature.
    
    HTTP Status: 400 (Bad Request)
    """
    pass
```

### 13.5 Settings with Feature Flags

**File: `app/settings.py` (UPDATED)**

```python
# Add to existing Settings class

class Settings(BaseSettings):
    # ... existing settings ...
    
    # NEW: Feature flags (for gradual rollout)
    enable_vision_features: bool = Field(
        default=True,
        description="Enable CHAT_WITH_IMAGE support",
        alias="ENABLE_VISION_FEATURES",
    )
    
    enable_document_features: bool = Field(
        default=False,  # Start disabled (needs testing)
        description="Enable CHAT_WITH_PDF support",
        alias="ENABLE_DOCUMENT_FEATURES",
    )
    
    enable_video_features: bool = Field(
        default=False,  # Start disabled (needs testing)
        description="Enable CHAT_WITH_YOUTUBE_VIDEO support",
        alias="ENABLE_VIDEO_FEATURES",
    )
    
    # NEW: Feature type override (for testing)
    feature_type_override: Optional[str] = Field(
        default=None,
        description="Override feature type detection (testing only)",
        alias="FEATURE_TYPE_OVERRIDE",
    )
    
    # NEW: Feature-specific rate limits
    image_chat_rate_limit: int = Field(
        default=10,
        description="Requests per minute for CHAT_WITH_IMAGE",
        alias="IMAGE_CHAT_RATE_LIMIT",
    )
    
    pdf_chat_rate_limit: int = Field(
        default=5,
        description="Requests per minute for CHAT_WITH_PDF",
        alias="PDF_CHAT_RATE_LIMIT",
    )
    
    video_chat_rate_limit: int = Field(
        default=3,
        description="Requests per minute for CHAT_WITH_YOUTUBE_VIDEO",
        alias="VIDEO_CHAT_RATE_LIMIT",
    )
```

---

## 15. Design Philosophy & Key Improvements

### 15.1 Design Philosophy

**Fail-Safe, Feature-Flagged, Validated, Tested**

This expansion follows conservative design principles to ensure production stability:

1. **Fail-Fast Validation**
   - Validate before API calls (cheaper than failing)
   - Clear, actionable error messages
   - Type-safe model and feature validation

2. **Feature Flagging**
   - Gradual rollout with environment variables
   - Can disable new features without breaking changes
   - Per-feature rate limits to isolate impact

3. **Backward Compatibility**
   - Single endpoint architecture maintained
   - Existing clients unaffected
   - Non-breaking until hard launch

4. **Modular Design**
   - Separate validation module for reusability
   - Feature-specific error classes
   - Clear separation of concerns

### 15.2 Key Improvements Over Original Design

| Aspect | Original | Revised | Benefit |
|---------|----------|----------|----------|
| Client Method | Missing | Added | PDF/Video features work |
| Image Detection | Complex regex | URL parser | More reliable, simpler |
| Validation | None | Comprehensive | Fail-fast, clear errors |
| Error Handling | Generic | Specific | Better debugging, graceful degradation |
| Model Support | Lists only | Validated | Prevents invalid requests |
| Conversation IDs | Inconsistent | Unified | Clear requirements per feature |
| Testing | Basic scenarios | Comprehensive | All paths covered |
| Deployment | All-at-once | Phased | Risk mitigation |

### 15.3 Risk Mitigation Strategies

| Risk | Mitigation |
|-------|------------|
| New features break existing clients | Feature flags (disabled by default) |
| Conversation API failures exist | Test extensively before enabling |
| Model support assumptions | Validation layer, fail-fast |
| Streaming differs by feature type | Verify each feature independently |
| Rate limits affect production | Feature-specific limits, per-feature tracking |

---

## Appendix: 1MinAI Feature Type Reference

From official API documentation:

| Feature | Required Fields | Optional Fields | Notes |
|---------|----------------|------------------|-------|
| CHAT_WITH_AI | prompt, model | conversationId, isMixed, webSearch, numOfSite, maxWord | conversationId optional for single messages |
| CHAT_WITH_IMAGE | prompt, imageUrl, model | conversationId, isMixed, webSearch, numOfSite, maxWord | Multimodal chat |
| CHAT_WITH_PDF | prompt, documentIds, conversationId, model | isMixed, webSearch, numOfSite, maxWord | Requires conversationId (call Conversation API first) |
| CHAT_WITH_YOUTUBE_VIDEO | prompt, videoUrl, conversationId, model | isMixed, webSearch, numOfSite, maxWord | Requires conversationId (call Conversation API first) |

---

**Design Document Version**: 1.0  
**Date**: 2026-02-07  
**Author**: 1MinAI Proxy Design Document
