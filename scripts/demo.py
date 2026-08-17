import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.service import GameDevCopilotService


CASES = [
    ("game", "1.3版本Android切换装备时闪退，错误码E-EQP-500，应该怎么排查？"),
    ("game", "1.3版本MATCH-408后按钮无法再次点击"),
    ("game", "Android切换装备闪退 E-EQP-500，请创建Bug工单"),
    ("enterprise", "2.4版本订单接口连续500并出现DB-POOL-503，应该怎么排查？"),
]


def main() -> None:
    service = GameDevCopilotService(Settings())
    service.load_or_ingest()
    for index, (domain, message) in enumerate(CASES, start=1):
        result = service.diagnose(
            message=message, domain=domain, session_id=f"cli-demo-{index}"
        )
        print("=" * 70)
        print(f"工作空间：{domain} | 问题：{message}")
        print(result["answer"])
        print("执行轨迹：", " -> ".join(item["node"] for item in result["trace"]))


if __name__ == "__main__":
    main()
