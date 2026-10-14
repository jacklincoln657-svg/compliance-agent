"""一键跑完全部冒烟（请在项目根目录执行：python smokes/run_all.py）。"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "_smoke_freeze_scenarios.py",
    "_smoke_policy_ctx.py",
    "_smoke_policy_refs.py",
    "_smoke_compliance.py",
]


def main() -> None:
    # 保证子脚本能 import _bootstrap / 项目根模块
    sys.path.insert(0, str(HERE))
    root = HERE.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    for name in SCRIPTS:
        path = HERE / name
        print(f"\n======== {name} ========")
        runpy.run_path(str(path), run_name="__main__")
    print("\n=== ALL SMOKES PASSED ===")


if __name__ == "__main__":
    main()
