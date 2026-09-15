"""本地 `.env` 文件的支持（可选，无第三方依赖）。

**为什么需要**：`.env.example` 一直告诉使用者"复制为 `.env` 并填写"，但代码只读环境变量、
从不读这个文件——说明与实际行为不符。这里补上最小实现（不引入 python-dotenv）：

* 只在**环境变量尚未设置**时填入，因此真实环境变量优先级更高；
* 不覆盖已有值，不写入 os.environ 之外的任何地方；
* 语法只支持 ``KEY=VALUE``、``#`` 注释、可选引号与 ``export `` 前缀——够用即可。

**安全**：`.env` 在 `.gitignore` 中，不会被提交；本模块只负责读取，不做任何上传。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PROJECT_ROOT", "load_env_file"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, _, value = stripped.partition("=")
    key = key.strip()
    value = value.strip()
    # 去掉成对引号（单引号/双引号），不做转义处理——够用即可，且避免意外的解释
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """把 ``.env`` 里的键值填进 ``os.environ``（**不覆盖**已存在的变量）。

    返回实际填入的键值对，便于调用方报告"读了哪些配置"。文件不存在时返回空字典
    （而不是报错）——`.env` 本质是可选的本地配置。
    """
    candidate = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not candidate.is_file():
        return {}
    applied: dict[str, str] = {}
    for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if not value:
            # 空值不覆盖：.env.example 里大量留空项，若写入空串会把默认值顶掉
            continue
        if os.environ.get(key):
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
