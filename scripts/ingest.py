import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.service import IncidentCopilotService


def main() -> None:
    service = IncidentCopilotService(Settings())
    try:
        result = service.ingest()
        print(result["message"])
        print(f"文档数量：{result['document_count']}")
        print(f"新增文本块：{result['new_chunk_count']}")
        print(f"检索引擎：{result['retrieval']}")
    finally:
        service.close()


if __name__ == "__main__":
    main()
