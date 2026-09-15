"""`.env` 支持与 `doctor` 子命令的测试（离线）。

守三件事：

① **`.env` 真的会被读取**：`.env.example` 长期承诺"复制为 `.env` 后填写"，
   而代码以前只读环境变量——说明与实际行为不一致，这类落差必须有测试盯着；
② **环境变量优先**：`.env` 只在变量未设置时填入，否则 CI 与显式 export 会被本地文件顶掉；
③ **不泄露密钥**：`doctor` 只报告来源与长度，绝不能打印密钥内容。
"""

import os
from pathlib import Path

from aqlab.cli import build_parser
from aqlab.env_file import load_env_file


def test_load_env_file_parses_common_syntax(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# 注释行",
                "",
                "AQLAB_TEST_PLAIN=abc",
                'AQLAB_TEST_QUOTED="a b c"',
                "AQLAB_TEST_SINGLE='x=y'",
                "export AQLAB_TEST_EXPORTED=1",
                "AQLAB_TEST_EMPTY=",
                "NO_EQUALS_SIGN",
            ]
        ),
        encoding="utf-8",
    )
    applied = load_env_file(env)
    assert applied["AQLAB_TEST_PLAIN"] == "abc"
    assert applied["AQLAB_TEST_QUOTED"] == "a b c"
    assert applied["AQLAB_TEST_SINGLE"] == "x=y"
    assert applied["AQLAB_TEST_EXPORTED"] == "1"
    # 空值不写入：.env.example 里大量留空项，若写入空串会把默认值顶掉
    assert "AQLAB_TEST_EMPTY" not in applied
    assert os.environ.get("AQLAB_TEST_EMPTY") is None

    for key in ("AQLAB_TEST_PLAIN", "AQLAB_TEST_QUOTED", "AQLAB_TEST_SINGLE", "AQLAB_TEST_EXPORTED"):
        os.environ.pop(key, None)


def test_existing_environment_wins(tmp_path, monkeypatch):
    """已设置的环境变量优先级更高，`.env` 不得覆盖它。"""
    monkeypatch.setenv("AQLAB_TEST_PRIORITY", "from-environment")
    env = tmp_path / ".env"
    env.write_text("AQLAB_TEST_PRIORITY=from-file\nAQLAB_TEST_FRESH=1\n", encoding="utf-8")
    applied = load_env_file(env)
    assert os.environ["AQLAB_TEST_PRIORITY"] == "from-environment"
    assert "AQLAB_TEST_PRIORITY" not in applied
    assert applied["AQLAB_TEST_FRESH"] == "1"
    os.environ.pop("AQLAB_TEST_FRESH", None)


def test_missing_env_file_is_not_an_error(tmp_path):
    """`.env` 是可选的本地配置，缺失时返回空字典而不是抛异常。"""
    assert load_env_file(tmp_path / "does-not-exist") == {}
    directory = tmp_path / "a-directory"
    directory.mkdir()
    assert load_env_file(directory) == {}


def test_env_example_documents_every_key_the_code_reads():
    """`.env.example` 必须覆盖代码里读取的所有 AQLAB_/TUSHARE_/FEISHU_ 变量。

    这条防的是"代码加了新变量、示例文件没跟上"，使用者照着示例填却发现漏了。
    """
    root = Path(__file__).resolve().parents[1]
    example = (root / ".env.example").read_text(encoding="utf-8")
    documented = set()
    for line in example.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            documented.add(stripped.split("=", 1)[0].strip())

    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "src" / "aqlab").glob("*.py")
    )
    import re

    read_keys = set(re.findall(r'environ\.get\("([A-Z][A-Z0-9_]+)"', source))
    read_keys |= set(re.findall(r'environ\["([A-Z][A-Z0-9_]+)"\]', source))
    # 只关心本项目的配置前缀，避免把第三方变量算进来
    relevant = {key for key in read_keys if key.startswith(("AQLAB_", "TUSHARE_", "FEISHU_"))}
    missing = sorted(relevant - documented)
    assert not missing, f".env.example 缺少这些变量：{missing}"


def test_doctor_command_exists_and_never_prints_the_key(capsys, monkeypatch):
    """doctor 必须存在，且只报告来源与长度——绝不能把密钥打出来。"""
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert "doctor" in choices, "应提供 doctor 子命令"

    secret = "sk-super-secret-value-should-never-be-printed"
    monkeypatch.setenv("AQLAB_LLM_API_KEY", secret)
    args = parser.parse_args(["doctor"])
    assert args.func(args) == 0
    captured = capsys.readouterr().out
    assert secret not in captured, "doctor 不得打印密钥内容"
    assert "长度" in captured or "长度" in captured
