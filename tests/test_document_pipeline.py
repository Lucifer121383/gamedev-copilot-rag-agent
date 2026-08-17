import json
from pathlib import Path

from app.chunker import chunk_sections, split_text
from app.document_loader import load_document


def test_json_bug_records_keep_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bugs.json"
    path.write_text(
        json.dumps(
            [{"bug_id": "BUG-1", "module": "装备系统", "root_cause": "空资源"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sections = load_document(path, domain="game", doc_type="bug_tickets")
    assert sections[0].metadata["bug_id"] == "BUG-1"
    assert "root_cause: 空资源" in sections[0].text


def test_csv_test_cases_are_individual_sections(tmp_path: Path) -> None:
    path = tmp_path / "cases.csv"
    path.write_text(
        "case_id,module,expected\nTC-1,装备系统,不得崩溃\nTC-2,登录系统,登录成功\n",
        encoding="utf-8",
    )
    sections = load_document(path, domain="game", doc_type="test_cases")
    assert len(sections) == 2
    assert sections[1].section == "TC-2"


def test_chunking_preserves_domain_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bug.json"
    path.write_text(
        '[{"bug_id":"BUG-9","text":"装备切换时资源句柄为空导致崩溃"}]',
        encoding="utf-8",
    )
    sections = load_document(path, domain="game", doc_type="bug_tickets")
    chunks = chunk_sections(sections, chunk_size=100, overlap=20)
    assert chunks[0].domain == "game"
    assert chunks[0].doc_type == "bug_tickets"
    assert chunks[0].metadata["bug_id"] == "BUG-9"


def test_split_text_rejects_dead_loop_configuration() -> None:
    try:
        split_text("abcdef", chunk_size=4, overlap=4)
    except ValueError:
        pass
    else:
        raise AssertionError("overlap等于chunk_size时必须报错")

