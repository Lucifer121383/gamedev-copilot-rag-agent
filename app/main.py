from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.database import VALID_TICKET_STATUSES
from app.document_loader import SUPPORTED_SUFFIXES
from app.models import VALID_DOMAINS
from app.observability import configure_logging, request_id_context
from app.schemas import (
    ApprovalRequest,
    DiagnoseRequest,
    DiagnoseResponse,
    IndexJobItem,
    SearchRequest,
    TicketItem,
    TicketStatusUpdate,
)
from app.service import IncidentCopilotService


settings = Settings()
configure_logging(settings.log_level, settings.json_logs)
logger = logging.getLogger("incident_copilot.api")
service = IncidentCopilotService(settings)


class FixedWindowLimiter:
    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            bucket = self._requests[key]
            while bucket and now - bucket[0] >= self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


limiter = FixedWindowLimiter(settings.rate_limit_per_minute)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(service.load_or_ingest)
    try:
        yield
    finally:
        await run_in_threadpool(service.close)


app = FastAPI(
    title="IncidentCopilot：企业研发故障诊断与处置RAG Agent",
    description=(
        "支持游戏研发与企业软件双场景、BM25/BGE/FAISS混合检索、Rerank、"
        "LangGraph工作流、Function Calling、人工审批、幂等工单和离线评测。"
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next: Any):
    request_id = request.headers.get("X-Request-ID") or f"http-{uuid4().hex[:14]}"
    token = request_id_context.set(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        client_key = request.client.host if request.client else "unknown"
        if request.url.path.startswith("/api/") and not await limiter.allow(client_key):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": "请求过于频繁，请稍后重试",
                    },
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id, "Retry-After": "60"},
            )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response
    finally:
        request_id_context.reset(token)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "请求参数不符合要求",
                "details": exc.errors(),
            },
            "request_id": getattr(request.state, "request_id", "-"),
        },
    )


@app.exception_handler(FastAPIHTTPException)
async def http_error(request: Request, exc: FastAPIHTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "请求处理失败"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"code": "http_error", "message": message, "details": exc.detail},
            "request_id": getattr(request.state, "request_id", "-"),
        },
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error")
    return JSONResponse(
        status_code=500,
        content={
            "error": {"code": "internal_error", "message": "服务内部错误"},
            "request_id": getattr(request.state, "request_id", "-"),
        },
    )


def require_api_key(request: Request) -> None:
    if not settings.api_auth_enabled:
        return
    provided = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(status_code=401, detail="API Key无效或缺失")


web_dir = settings.project_root / "web"
app.mount("/static", StaticFiles(directory=web_dir), name="static")
api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/healthz", include_in_schema=False)
def public_health() -> dict[str, str]:
    return {"status": "ok", "service": "IncidentCopilot"}


@api.get("/health")
def health() -> dict:
    return {"status": "ok", **service.stats()}


@api.get("/stats")
def stats() -> dict:
    return service.stats()


@api.get("/workspaces")
def workspaces() -> list[dict[str, str]]:
    return service.workspaces()


@api.get("/tools")
def tools() -> list[dict]:
    return service.tool_schemas()


@api.post("/ingest")
async def ingest(domain: str | None = None) -> dict:
    if domain is not None and domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail="不支持的工作空间")
    try:
        return await run_in_threadpool(service.ingest, domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/index-jobs", response_model=IndexJobItem)
async def start_index_job(domain: str | None = None) -> dict[str, Any]:
    try:
        return await run_in_threadpool(service.start_index_job, domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.get("/index-jobs/{job_id}", response_model=IndexJobItem)
async def get_index_job(job_id: str) -> dict[str, Any]:
    job = await run_in_threadpool(service.store.get_index_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="索引任务不存在")
    return job


@api.post("/upload/{domain}")
async def upload(domain: str, files: list[UploadFile] = File(...)) -> dict:
    if domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail="不支持的工作空间")
    saved: list[str] = []
    rejected: list[str] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024
    upload_dir = settings.upload_dir(domain)
    for upload_file in files:
        raw_name = (upload_file.filename or "unnamed").replace("\\", "/")
        safe_name = Path(raw_name).name
        if not safe_name or safe_name in {".", ".."}:
            rejected.append("无效文件名")
            continue
        if Path(safe_name).suffix.lower() not in SUPPORTED_SUFFIXES:
            rejected.append(f"{safe_name}: 不支持的格式")
            continue
        content = await upload_file.read(max_bytes + 1)
        if not content:
            rejected.append(f"{safe_name}: 空文件")
            continue
        if len(content) > max_bytes:
            rejected.append(f"{safe_name}: 超过{settings.max_upload_mb}MB")
            continue
        target = upload_dir / safe_name
        if target.exists():
            target = upload_dir / f"{target.stem}-{uuid4().hex[:8]}{target.suffix}"
        target.write_bytes(content)
        saved.append(target.name)
    return {"domain": domain, "saved": saved, "rejected": rejected}


@api.post("/search")
async def search(request: SearchRequest) -> dict:
    hits = await run_in_threadpool(
        service.search,
        request.query,
        domain=request.domain,
        top_k=request.top_k,
        filters=request.filters,
    )
    return {
        "query": request.query,
        "domain": request.domain,
        "retrieval": service.retriever.describe(),
        "sources": [
            service._source_payload(hit, index) for index, hit in enumerate(hits, start=1)
        ],
    }


@api.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest) -> dict:
    try:
        return await run_in_threadpool(
            service.diagnose,
            message=request.message,
            domain=request.domain,
            session_id=request.session_id,
            top_k=request.top_k,
            idempotency_key=request.idempotency_key,
            approve_write=request.approve_write,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/diagnose/stream")
async def diagnose_stream(request: DiagnoseRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        yield f"event: status\ndata: {json.dumps({'node': 'intake', 'message': '开始处理'}, ensure_ascii=False)}\n\n"
        try:
            result = await run_in_threadpool(
                service.diagnose,
                message=request.message,
                domain=request.domain,
                session_id=request.session_id,
                top_k=request.top_k,
                idempotency_key=request.idempotency_key,
                approve_write=request.approve_write,
            )
            for trace in result["trace"]:
                yield f"event: trace\ndata: {json.dumps(trace, ensure_ascii=False)}\n\n"
            answer = result.get("answer", "")
            for start in range(0, len(answer), 24):
                yield f"event: token\ndata: {json.dumps({'text': answer[start:start + 24]}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            yield f"event: result\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"
        except Exception as exc:
            payload = {"code": "stream_error", "message": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@api.post("/approvals/{request_id}", response_model=DiagnoseResponse)
async def approve(request_id: str, request: ApprovalRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            service.approve,
            request_id,
            approved=request.approved,
            reason=request.reason,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.get("/agent-runs/{request_id}")
async def agent_run(request_id: str) -> dict[str, Any]:
    run = await run_in_threadpool(service.store.get_agent_run, request_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent请求不存在")
    return run


@api.get("/sessions/{session_id}/messages")
async def messages(session_id: str, limit: int = 50) -> dict:
    return {
        "session_id": session_id,
        "messages": await run_in_threadpool(service.store.get_messages, session_id, limit),
    }


@api.get("/tickets", response_model=list[TicketItem])
async def list_tickets(
    domain: str | None = None, status: str | None = None, limit: int = 20
) -> list[dict]:
    if domain is not None and domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail="不支持的工作空间")
    if status is not None and status not in VALID_TICKET_STATUSES:
        raise HTTPException(status_code=400, detail="不支持的工单状态")
    return await run_in_threadpool(service.store.list_tickets, domain, status, limit)


@api.get("/tickets/{ticket_no}", response_model=TicketItem)
async def get_ticket(ticket_no: str) -> dict:
    ticket = await run_in_threadpool(service.store.get_ticket, ticket_no)
    if ticket is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    return ticket


@api.patch("/tickets/{ticket_no}", response_model=TicketItem)
async def update_ticket(ticket_no: str, request: TicketStatusUpdate) -> dict:
    ticket = await run_in_threadpool(
        service.store.update_ticket_status, ticket_no, request.status
    )
    if ticket is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    return ticket


app.include_router(api)
