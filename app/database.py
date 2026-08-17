from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models import VALID_DOMAINS


VALID_TICKET_STATUSES = {"open", "processing", "resolved", "closed"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CopilotStore:
    """会话、Agent轨迹与幂等Bug工单的SQLite存储。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL CHECK(domain IN ('game', 'enterprise')),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_no TEXT UNIQUE,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    domain TEXT NOT NULL CHECK(domain IN ('game', 'enterprise')),
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL CHECK(severity IN ('P1', 'P2', 'P3')),
                    platform TEXT NOT NULL,
                    module TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open'
                        CHECK(status IN ('open', 'processing', 'resolved', 'closed')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS agent_runs (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_tickets_domain_status
                    ON tickets(domain, status, id);
                """
            )

    def ensure_session(self, session_id: str, domain: str) -> None:
        if domain not in VALID_DOMAINS:
            raise ValueError("无效工作空间")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT domain FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row and row["domain"] != domain:
                raise ValueError("同一会话不能跨工作空间使用")
            connection.execute(
                "INSERT OR IGNORE INTO sessions(session_id, domain, created_at) VALUES (?, ?, ?)",
                (session_id, domain, _now()),
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
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, created_at),
            )
        return {
            "id": cursor.lastrowid,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": created_at,
        }

    def get_messages(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM messages WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
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
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT ticket_no FROM tickets WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if existing:
            ticket = self.get_ticket(existing["ticket_no"])
            assert ticket is not None
            return {**ticket, "cached": True}

        timestamp = _now()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO tickets(
                        idempotency_key, session_id, domain, category, severity,
                        platform, module, title, description, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
                    """,
                    (
                        idempotency_key,
                        session_id,
                        domain,
                        category,
                        severity,
                        platform,
                        module,
                        title[:120],
                        description,
                        timestamp,
                        timestamp,
                    ),
                )
                ticket_id = int(cursor.lastrowid)
                prefix = "GME" if domain == "game" else "ENT"
                date_part = datetime.now(UTC).strftime("%Y%m%d")
                ticket_no = f"{prefix}-{date_part}-{ticket_id:05d}"
                connection.execute(
                    "UPDATE tickets SET ticket_no = ? WHERE id = ?", (ticket_no, ticket_id)
                )
        except sqlite3.IntegrityError:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT ticket_no FROM tickets WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
            if not row:
                raise
            ticket_no = row["ticket_no"]
            ticket = self.get_ticket(ticket_no)
            assert ticket is not None
            return {**ticket, "cached": True}

        ticket = self.get_ticket(ticket_no)
        assert ticket is not None
        return {**ticket, "cached": False}

    def get_ticket(self, ticket_no: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT ticket_no, idempotency_key, session_id, domain, category,
                       severity, platform, module, title, description, status,
                       created_at, updated_at
                FROM tickets WHERE ticket_no = ?
                """,
                (ticket_no,),
            ).fetchone()
        return dict(row) if row else None

    def list_tickets(
        self, domain: str | None = None, status: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        clauses: list[str] = []
        parameters: list[Any] = []
        if domain:
            clauses.append("domain = ?")
            parameters.append(domain)
        if status:
            clauses.append("status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT ticket_no, idempotency_key, session_id, domain, category,
                       severity, platform, module, title, description, status,
                       created_at, updated_at
                FROM tickets {where} ORDER BY id DESC LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def update_ticket_status(self, ticket_no: str, status: str) -> dict[str, Any] | None:
        if status not in VALID_TICKET_STATUSES:
            raise ValueError(f"不支持的工单状态: {status}")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_no = ?",
                (status, _now(), ticket_no),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_ticket(ticket_no)

    def record_agent_run(
        self,
        request_id: str,
        session_id: str,
        domain: str,
        trace: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        self.ensure_session(session_id, domain)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO agent_runs(
                    request_id, session_id, domain, trace_json, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    session_id,
                    domain,
                    json.dumps(trace, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                    _now(),
                ),
            )

    def ticket_stats(self, domain: str | None = None) -> dict[str, int]:
        counts = {status: 0 for status in VALID_TICKET_STATUSES}
        where = "WHERE domain = ?" if domain else ""
        parameters = (domain,) if domain else ()
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT status, COUNT(*) AS count FROM tickets {where} GROUP BY status",
                parameters,
            ).fetchall()
            total = connection.execute(
                f"SELECT COUNT(*) FROM tickets {where}", parameters
            ).fetchone()[0]
        counts.update({row["status"]: row["count"] for row in rows})
        return {"total": total, **counts}

