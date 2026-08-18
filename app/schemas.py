from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Domain = Literal["game", "enterprise"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    domain: Domain
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, str] | None = None


class DiagnoseRequest(BaseModel):
    message: str = Field(min_length=2, max_length=3000)
    domain: Domain
    session_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    top_k: int = Field(default=5, ge=1, le=10)
    idempotency_key: str | None = Field(
        default=None,
        min_length=8,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_.:-]+$",
    )
    approve_write: bool = False


class ApprovalRequest(BaseModel):
    approved: bool
    reason: str | None = Field(default=None, max_length=500)


class SourceItem(BaseModel):
    citation: str
    source: str
    domain: Domain
    doc_type: str
    page: int | None = None
    section: str | None = None
    score: float
    score_breakdown: dict[str, float]
    metadata: dict[str, Any]
    text: str


class DiagnosisItem(BaseModel):
    category: str
    severity: Literal["P1", "P2", "P3"]
    platform: str
    module: str
    version: str | None = None
    error_code: str | None = None
    possible_causes: list[str]
    reproduction_steps: list[str]
    regression_tests: list[str]
    confidence: float
    needs_human_review: bool


class EvidenceItem(BaseModel):
    sufficient: bool
    confidence: float
    reason: str
    source_count: int


class TraceItem(BaseModel):
    step: int
    node: str
    status: str
    detail: str


class RequestMetrics(BaseModel):
    retrieval_ms: float
    generation_ms: float
    active_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_retries: int
    citation_valid: bool


class TicketItem(BaseModel):
    ticket_no: str
    idempotency_key: str
    session_id: str
    domain: Domain
    category: str
    severity: Literal["P1", "P2", "P3"]
    platform: str
    module: str
    title: str
    description: str
    status: Literal["open", "processing", "resolved", "closed"]
    created_at: str
    updated_at: str
    cached: bool | None = None


class DiagnoseResponse(BaseModel):
    request_id: str
    session_id: str
    domain: Domain
    message: str
    status: Literal["completed", "awaiting_approval"]
    answer: str
    mode: str
    warning: str | None = None
    action: Literal[
        "answer", "human_review", "awaiting_approval", "create_ticket", "write_rejected"
    ]
    diagnosis: DiagnosisItem
    evidence: EvidenceItem
    sources: list[SourceItem]
    tool_results: list[dict[str, Any]]
    ticket: TicketItem | None = None
    approval: dict[str, Any] | None = None
    planner_mode: str
    trace: list[TraceItem]
    metrics: RequestMetrics


class TicketStatusUpdate(BaseModel):
    status: Literal["open", "processing", "resolved", "closed"]


class IndexJobItem(BaseModel):
    job_id: str
    domain: str | None
    status: Literal["queued", "running", "completed", "failed"]
    progress: int
    message: str
    result: dict[str, Any] | None = None
    created_at: str
    updated_at: str
