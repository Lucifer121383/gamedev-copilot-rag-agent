from pathlib import Path

import pytest

from app.config import Settings
from app.service import GameDevCopilotService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def service(tmp_path: Path) -> GameDevCopilotService:
    settings = Settings(
        project_root=PROJECT_ROOT,
        data_dir=PROJECT_ROOT / "data",
        storage_dir=tmp_path / "storage",
        embedding_backend="hashing",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    instance = GameDevCopilotService(settings)
    instance.ingest()
    yield instance
    instance.close()
