from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    system_prompt: Optional[str] = "You are a helpful AI assistant."


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CostDetails(BaseModel):
    input_cost: float
    output_cost: float
    total_cost: float


class ChatResponse(BaseModel):
    reply: str
    model: str
    duration_seconds: float
    usage: TokenUsage
    cost: CostDetails
    from_cache: bool