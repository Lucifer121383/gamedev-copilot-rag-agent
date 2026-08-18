"""IncidentCopilot本地启动入口。"""

from uvicorn.config import Config
from uvicorn.server import Server


if __name__ == "__main__":
    Server(
        Config("app.main:app", host="127.0.0.1", port=8010, reload=False)
    ).run()
