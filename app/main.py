from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.database import VALID_TICKET_STATUSES
from app.document_loader import SUPPORTED_SUFFIXES
from app.models import VALID_DOMAINS
from app.schemas import (
    DiagnoseRequest,
    DiagnoseResponse,
    SearchRequest,
    TicketItem,
    TicketStatusUpdate,
)
from app.service import GameDevCopilotService


settings = Settings()
service = GameDevCopilotService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(service.load_or_ingest)
    yield


app = FastAPI(
    title="GameDev Copilot：研发测试与故障诊断RAG Agent",
    description=(
        "支持游戏与企业软件双场景、混合检索、结构化Bug诊断、"
        "Function Calling工具、幂等工单和可观察Agent轨迹。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

web_dir = settings.project_root / "web"
app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", **service.stats()}


@app.get("/api/stats")
def stats() -> dict:
    return service.stats()


@app.get("/api/workspaces")
def workspaces() -> list[dict[str, str]]:
    return service.workspaces()


@app.get("/api/tools")
def tools() -> list[dict]:
    return service.tool_schemas()


@app.post("/api/ingest")
async def ingest(domain: str | None = None) -> dict:
    if domain is not None and domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail="不支持的工作空间")
    try:
        return await run_in_threadpool(service.ingest, domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/upload/{domain}")
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
        (upload_dir / safe_name).write_bytes(content)
        saved.append(safe_name)
    return {"domain": domain, "saved": saved, "rejected": rejected}


@app.post("/api/search")
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
        "sources": [
            service._source_payload(hit, index)
            for index, hit in enumerate(hits, start=1)
        ],
    }


@app.post("/api/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest) -> dict:
    try:
        return await run_in_threadpool(
            service.diagnose,
            message=request.message,
            domain=request.domain,
            session_id=request.session_id,
            top_k=request.top_k,
            idempotency_key=request.idempotency_key,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/messages")
async def messages(session_id: str, limit: int = 50) -> dict:
    return {
        "session_id": session_id,
        "messages": await run_in_threadpool(service.store.get_messages, session_id, limit),
    }


@app.get("/api/tickets", response_model=list[TicketItem])
async def list_tickets(
    domain: str | None = None, status: str | None = None, limit: int = 20
) -> list[dict]:
    if domain is not None and domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail="不支持的工作空间")
    if status is not None and status not in VALID_TICKET_STATUSES:
        raise HTTPException(status_code=400, detail="不支持的工单状态")
    return await run_in_threadpool(service.store.list_tickets, domain, status, limit)


@app.get("/api/tickets/{ticket_no}", response_model=TicketItem)
async def get_ticket(ticket_no: str) -> dict:
    ticket = await run_in_threadpool(service.store.get_ticket, ticket_no)
    if ticket is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    return ticket


@app.patch("/api/tickets/{ticket_no}", response_model=TicketItem)
async def update_ticket(ticket_no: str, request: TicketStatusUpdate) -> dict:
    ticket = await run_in_threadpool(
        service.store.update_ticket_status, ticket_no, request.status
    )
    if ticket is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    return ticket

