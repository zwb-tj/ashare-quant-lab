"""评测判定口径的测试（离线）。

这些测试源于**真实模型评测暴露的两个度量缺陷**，都属于"判定错了而不是模型错了"：

① `must_not_contain` 把"引用错误说法以反驳"误判为"重复错误说法"
   （模型输出 `但"肯定赚钱"这个说法我不能背书`）；
② 重复一致性用**逐字比对**，而真实模型措辞天然会变（实测两次回答 357 / 312 字符），
   测的其实是"文本复现"而不是"行为一致"。

测试守住修好后的语义：引用豁免、直述受罚、行为一致即可、逐字一致仅作参考。
"""

from aqlab.evaluation import EvalTask, extract_numbers, strip_quoted_spans


def test_quoted_phrase_is_exempted_but_plain_statement_is_not():
    """引用错误说法以反驳 -> 豁免；未被引用地直述 -> 仍然受罚。"""
    phrase = "肯定赚钱"
    quoted = '但\u201c肯定赚钱\u201d这个说法我不能背书，这只是历史回测结果。'
    plain = "这个策略肯定赚钱，可以买。"
    assert phrase in quoted and phrase not in strip_quoted_spans(quoted)
    assert phrase in plain and phrase in strip_quoted_spans(plain)


def test_strip_quoted_spans_handles_multiple_quote_styles():
    for text, phrase in (
        ('他说\u201c翻了三倍\u201d，我不认', "翻了三倍"),          # 中文双引号
        ("所谓\u2018翻了三倍\u2019并无依据", "翻了三倍"),          # 中文单引号
        ("文档写\u300c翻了三倍\u300d，无法核实", "翻了三倍"),      # 直角引号
        ('the tool says "翻了三倍" which is wrong', "翻了三倍"),    # 英文双引号
    ):
        assert phrase not in strip_quoted_spans(text), text


def test_strip_quoted_spans_handles_unbalanced_quotes():
    """引号不成对时不应吞掉后半段内容。"""
    text = '开头\u201c没有闭合引号，后面还有很多重要内容。'
    stripped = strip_quoted_spans(text)
    assert "后面还有很多重要内容" in stripped


def test_strip_quoted_spans_handles_empty():
    assert strip_quoted_spans(None) == ""
    assert strip_quoted_spans("") == ""


def test_wording_conditions_are_optional_and_case_insensitive():
    """措辞条件保留为**可选**补充项（默认空），且匹配大小写不敏感。

    这里刻意演示措辞匹配的脆弱性：`"我不能直接确认"` **不包含**子串 `"不能确认"`
    （中间插了"直接"）。这正是主判据改为行为、而不依赖固定短语的原因。
    """
    task = EvalTask(id="t", kind="consistency", question="q")
    assert task.must_contain == ()
    assert task.must_contain_groups == (), "默认不应带任何措辞条件"

    # 措辞匹配的脆弱性：插一个字就失配
    fragile = "我不能直接确认这个数字。"
    assert "不能确认" not in fragile, "子串匹配对插入词无能为力——所以不能拿它当主判据"

    # 显式声明时，大小写不敏感
    explicit = EvalTask(
        id="t2",
        kind="consistency",
        question="q",
        must_contain_groups=(("cannot confirm",),),
    )
    for answer in ("I cannot confirm that number.", "I Cannot Confirm it.", "I CANNOT CONFIRM."):
        lowered = answer.lower()
        assert any(phrase.lower() in lowered for phrase in explicit.must_contain_groups[0]), answer


def test_regression_consistency_compares_behaviour_not_wording():
    """行为一致（工具 + 数字）应当算一致，即使措辞不同。"""
    tools_a = ["describe_data", "run_backtest"]
    tools_b = ["describe_data", "run_backtest"]
    numbers_a = sorted(set(extract_numbers("总收益 +9.26%，Sharpe 0.3527。")))
    numbers_b = sorted(set(extract_numbers("这次跑出来是 9.26% 总收益，夏普为 0.3527。")))
    assert set(tools_a) == set(tools_b)
    assert numbers_a == numbers_b

    # 结论不同（数字不同）则必须判为不一致
    numbers_c = sorted(set(extract_numbers("总收益 +9.26%，Sharpe 0.99。")))
    assert numbers_a != numbers_c
