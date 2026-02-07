"""
1minAI API request schemas based on the AI Feature API documentation.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ChatPromptObject(BaseModel):
    """PromptObject for CHAT_WITH_AI and CHAT_WITH_IMAGE feature types."""

    prompt: str = Field(..., description="The user's message/prompt")
    isMixed: bool = Field(default=False, description="Mix models context")
    imageList: Optional[list[str]] = Field(
        default=None, description="Asset keys for CHAT_WITH_IMAGE"
    )
    webSearch: bool = Field(default=False, description="Enable web search")
    numOfSite: int = Field(default=1, description="Number of sites to search")
    maxWord: int = Field(default=500, description="Maximum words from web search")


class OneMinAIChatRequest(BaseModel):
    """Request body for 1minAI CHAT_WITH_AI feature."""

    type: str = Field(default="CHAT_WITH_AI", description="Feature type")
    model: str = Field(..., description="Model name to use")
    promptObject: ChatPromptObject
    conversationId: Optional[str] = Field(
        default=None, description="Optional conversation ID for context"
    )
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")


class TeamUser(BaseModel):
    """Team user details from 1minAI response."""

    teamId: Optional[str] = None
    userId: Optional[str] = None
    userName: Optional[str] = None
    userAvatar: Optional[str] = None
    status: Optional[str] = None
    role: Optional[str] = None
    creditLimit: Optional[int] = None
    usedCredit: Optional[int] = None
    createdAt: Optional[str] = None
    createdBy: Optional[str] = None
    updatedAt: Optional[str] = None
    updatedBy: Optional[str] = None


class AIRecordDetail(BaseModel):
    """Detailed request and response data from 1minAI."""

    promptObject: Optional[dict] = None
    resultObject: Optional[list] = None


class AIRecord(BaseModel):
    """Main response object from 1minAI non-streaming endpoint."""

    uuid: Optional[str] = None
    userId: Optional[str] = None
    teamId: Optional[str] = None
    teamUser: Optional[TeamUser] = None
    model: Optional[str] = None
    type: Optional[str] = None
    metadata: Optional[dict] = None
    rating: Optional[int] = None
    feedback: Optional[str] = None
    conversationId: Optional[str] = None
    status: Optional[str] = None
    createdAt: Optional[str] = None
    aiRecordDetail: Optional[AIRecordDetail] = None
    additionalData: Optional[dict] = None


class OneMinAIChatResponse(BaseModel):
    """Non-streaming response from 1minAI API."""

    aiRecord: AIRecord
