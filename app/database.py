from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import IntegrityError

from app.models import VALID_DOMAINS


VALID_TICKET_STATUSES = {"open", "processing", "resolved", "closed"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


metadata = MetaData()

sessions = Table(
    "sessions",
    metadata,
    Column("session_id", String(100), primary_key=True),
    Column("domain", String(20), nullable=False),
    Column("created_at", String(40), nullable=False),
    CheckConstraint("domain IN ('game', 'enterprise')", name="ck_sessions_domain"),
)

messages = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(100), ForeignKey("sessions.session_id"), nullable=False),
    Column("role", String(20), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", String(40), nullable=False),
    CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
)
Index("idx_messages_session", messages.c.session_id, messages.c.id)

tickets = Table(
    "tickets",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticket_no", String(50), unique=True),
    Column("idempotency_key", String(100), nullable=False),
    Column("session_id", String(100), ForeignKey("sessions.session_id"), nullable=False),
    Column("domain", String(20), nullable=False),
    Column("category", String(80), nullable=False),
    Column("severity", String(10), nullable=False),
    Column("platform", String(80), nullable=False),
    Column("module", String(120), nullable=False),
    Column("title", String(160), nullable=False),
    Column("description", Text, nullable=False),
    Column("status", String(20), nullable=False, server_default="open"),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
    UniqueConstraint("idempotency_key", name="uq_tickets_idempotency_key"),
    CheckConstraint("domain IN ('game', 'enterprise')", name="ck_tickets_domain"),
    CheckConstraint("severity IN ('P1', 'P2', 'P3')", name="ck_tickets_severity"),
    CheckConstraint(
        "status IN ('open', 'processing', 'resolved', 'closed')",
        name="ck_tickets_status",
    ),
)
Index("idx_tickets_domain_status", tickets.c.domain, tickets.c.status, tickets.c.id)

agent_runs = Table(
    "agent_runs",
    metadata,
    Column("request_id", String(80), primary_key=True),
    Column("session_id", String(100), ForeignKey("sessions.session_id"), nullable=False),
    Column("domain", String(20), nullable=False),
    Column("status", String(30), nullable=False),
    Column("trace_json", Text, nullable=False),
    Column("result_json", Text, nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
)
Index("idx_agent_runs_session", agent_runs.c.session_id, agent_runs.c.created_at)

index_jobs = Table(
    "index_jobs",
    metadata,
    Column("job_id", String(80), primary_key=True),
    Column("domain", String(20)),
    Column("status", String(30), nullable=False),
    Column("progress", Integer, nullable=False, server_default="0"),
    Column("message", Text, nullable=False, server_default=""),
    Column("result_json", Text),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
)


def _mapping(row: RowMapping | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class CopilotStore:
    """SQLAlchemy持久化层，默认SQLite，也可通过DATABASE_URL切换PostgreSQL。"""

    def __init__(self, database: Path | str) -> None:
        if isinstance(database, Path):
            database.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{database.as_posix()}"
        else:
            database_url = database
            if database_url.startswith("sqlite:///"):
                raw_path = database_url.removeprefix("sqlite:///")
                Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        connect_args = (
            {"check_same_thread": False, "timeout": 10}
            if database_url.startswith("sqlite")
            else {}
        )
        self.engine: Engine = create_engine(
            database_url, future=True, pool_pre_ping=True, connect_args=connect_args
        )
        if database_url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._configure_sqlite)
        metadata.create_all(self.engine)

    @staticmethod
    def _configure_sqlite(dbapi_connection: Any, _: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.close()

    def ensure_session(self, session_id: str, domain: str) -> None:
        if domain not in VALID_DOMAINS:
            raise ValueError("无效工作空间")
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(sessions.c.domain).where(sessions.c.session_id == session_id)
            ).scalar_one_or_none()
            if existing and existing != domain:
                raise ValueError("同一会话不能跨工作空间使用")
            if not existing:
                connection.execute(
                    insert(sessions).values(
                        session_id=session_id, domain=domain, created_at=_now()
                    )
                )

    def add_message(
        self, session_id: str, domain: str, role: str, content: str
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("role只允许user或assistant")
        content = content.strip()
        if not content:
            raise ValueError("消息不能为空")
        self.ensure_session(session_id, domain)
        created_at = _now()
        with self.engine.begin() as connection:
            result = connection.execute(
                insert(messages).values(
                    session_id=session_id,
                    role=role,
                    content=content,
                    created_at=created_at,
                )
            )
            message_id = int(result.inserted_primary_key[0])
        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": created_at,
        }

    def get_messages(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        statement = (
            select(messages)
            .where(messages.c.session_id == session_id)
            .order_by(messages.c.id.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in reversed(rows)]

    def create_ticket(
        self,
        *,
        idempotency_key: str,
        session_id: str,
        domain: str,
        category: str,
        severity: str,
        platform: str,
        module: str,
        title: str,
        description: str,
    ) -> dict[str, Any]:
        self.ensure_session(session_id, domain)
        existing = self._ticket_by_idempotency(idempotency_key)
        if existing:
            if existing["session_id"] != session_id or existing["domain"] != domain:
                raise ValueError("幂等键已被其他会话使用")
            return {**existing, "cached": True}

        timestamp = _now()
        ticket_no: str | None = None
        try:
            with self.engine.begin() as connection:
                result = connection.execute(
                    insert(tickets).values(
                        idempotency_key=idempotency_key,
                        session_id=session_id,
                        domain=domain,
                        category=category,
                        severity=severity,
                        platform=platform,
                        module=module,
                        title=title[:120],
                        description=description,
                        status="open",
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                )
                ticket_id = int(result.inserted_primary_key[0])
                prefix = "GME" if domain == "game" else "ENT"
                date_part = datetime.now(UTC).strftime("%Y%m%d")
                ticket_no = f"{prefix}-{date_part}-{ticket_id:05d}"
                connection.execute(
                    update(tickets)
                    .where(tickets.c.id == ticket_id)
                    .values(ticket_no=ticket_no)
                )
        except IntegrityError:
            existing = self._ticket_by_idempotency(idempotency_key)
            if existing is None:
                raise
            if existing["session_id"] != session_id or existing["domain"] != domain:
                raise ValueError("幂等键已被其他会话使用")
            return {**existing, "cached": True}

        assert ticket_no is not None
        ticket = self.get_ticket(ticket_no)
        assert ticket is not None
        return {**ticket, "cached": False}

    def _ticket_by_idempotency(self, key: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(tickets).where(tickets.c.idempotency_key == key)
            ).mappings().first()
        result = _mapping(row)
        if result:
            result.pop("id", None)
        return result

    def get_ticket(self, ticket_no: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(tickets).where(tickets.c.ticket_no == ticket_no)
            ).mappings().first()
        result = _mapping(row)
        if result:
            result.pop("id", None)
        return result

    def list_tickets(
        self, domain: str | None = None, status: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        statement = select(tickets)
        if domain:
            statement = statement.where(tickets.c.domain == domain)
        if status:
            statement = statement.where(tickets.c.status == status)
        statement = statement.order_by(tickets.c.id.desc()).limit(limit)
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        result = [dict(row) for row in rows]
        for item in result:
            item.pop("id", None)
        return result

    def update_ticket_status(self, ticket_no: str, status: str) -> dict[str, Any] | None:
        if status not in VALID_TICKET_STATUSES:
            raise ValueError(f"不支持的工单状态: {status}")
        with self.engine.begin() as connection:
            result = connection.execute(
                update(tickets)
                .where(tickets.c.ticket_no == ticket_no)
                .values(status=status, updated_at=_now())
            )
            if result.rowcount == 0:
                return None
        return self.get_ticket(ticket_no)

    def record_agent_run(
        self,
        request_id: str,
        session_id: str,
        domain: str,
        trace: list[dict[str, Any]],
        result: dict[str, Any],
        status: str = "completed",
    ) -> None:
        self.ensure_session(session_id, domain)
        timestamp = _now()
        values = {
            "request_id": request_id,
            "session_id": session_id,
            "domain": domain,
            "status": status,
            "trace_json": json.dumps(trace, ensure_ascii=False),
            "result_json": json.dumps(result, ensure_ascii=False),
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(agent_runs.c.request_id).where(agent_runs.c.request_id == request_id)
            ).scalar_one_or_none()
            if existing:
                values.pop("created_at")
                connection.execute(
                    update(agent_runs)
                    .where(agent_runs.c.request_id == request_id)
                    .values(**values)
                )
            else:
                connection.execute(insert(agent_runs).values(**values))

    def get_agent_run(self, request_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(agent_runs).where(agent_runs.c.request_id == request_id)
            ).mappings().first()
        raw = _mapping(row)
        if raw is None:
            return None
        raw["trace"] = json.loads(raw.pop("trace_json"))
        raw["result"] = json.loads(raw.pop("result_json"))
        return raw

    def ticket_stats(self, domain: str | None = None) -> dict[str, int]:
        counts = {status: 0 for status in VALID_TICKET_STATUSES}
        statement = select(tickets.c.status, func.count().label("count")).group_by(
            tickets.c.status
        )
        total_statement = select(func.count()).select_from(tickets)
        if domain:
            statement = statement.where(tickets.c.domain == domain)
            total_statement = total_statement.where(tickets.c.domain == domain)
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
            total = int(connection.execute(total_statement).scalar_one())
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        return {"total": total, **counts}

    def create_index_job(self, job_id: str, domain: str | None) -> dict[str, Any]:
        timestamp = _now()
        values = {
            "job_id": job_id,
            "domain": domain,
            "status": "queued",
            "progress": 0,
            "message": "等待处理",
            "result_json": None,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self.engine.begin() as connection:
            connection.execute(insert(index_jobs).values(**values))
        return values

    def update_index_job(
        self,
        job_id: str,
        *,
        status: str,
        progress: int,
        message: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(index_jobs)
                .where(index_jobs.c.job_id == job_id)
                .values(
                    status=status,
                    progress=max(0, min(progress, 100)),
                    message=message,
                    result_json=(json.dumps(result, ensure_ascii=False) if result else None),
                    updated_at=_now(),
                )
            )

    def get_index_job(self, job_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(index_jobs).where(index_jobs.c.job_id == job_id)
            ).mappings().first()
        raw = _mapping(row)
        if raw and raw.get("result_json"):
            raw["result"] = json.loads(raw.pop("result_json"))
        elif raw:
            raw.pop("result_json", None)
            raw["result"] = None
        return raw

    def clear_for_tests(self) -> None:
        with self.engine.begin() as connection:
            for table in (agent_runs, tickets, messages, sessions, index_jobs):
                connection.execute(delete(table))
