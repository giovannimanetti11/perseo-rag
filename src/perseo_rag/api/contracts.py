from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Question = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=4000,
    ),
]


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    collection_id: UUID | None = None
    limit: Annotated[int, Field(ge=1, le=50)] = 10


class CitationResponse(BaseModel):
    id: str
    source_ref: str


class QueryResponse(BaseModel):
    status: Literal["answered", "abstained"]
    text: str | None
    citations: tuple[CitationResponse, ...]
    abstention_reason: str | None
    evidence_count: int
    context_chars: int
    context_truncated: bool
