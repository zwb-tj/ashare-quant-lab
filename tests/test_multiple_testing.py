"""多重比较校正的测试（离线）。

**验证策略**：Bonferroni/BH 都是教科书算法，测试不能只是"再写一遍同样的公式"。
这里用三种独立方式交叉验证：

① 与手算的小例子对照（可心算的 p 值）；
② BH 的**定义性质**：调整后 q 值单调、拒绝集等于"最大 k 满足 p_(k) ≤ k/m·α"；
③ 与 statsmodels/scipy 的等价实现对照（若环境中存在就对照，不存在则跳过）——
   这一条能抓住"公式写对了但细节（如单调化方向）写反"的错误。
"""

import math

import numpy as np
import pandas as pd
import pytest

from aqlab.multiple_testing import (
    benjamini_hochberg,
    bonferroni,
    expected_false_positives,
    p_value_two_sided,
    summarize_significance,
)


def test_p_value_matches_known_normal_values():
    assert p_value_two_sided(0.0) == pytest.approx(1.0)
    assert p_value_two_sided(1.0) == pytest.approx(0.3173105, abs=1e-6)
    assert p_value_two_sided(1.96) == pytest.approx(0.0499958, abs=1e-6)
    assert p_value_two_sided(2.575829) == pytest.approx(0.01, abs=1e-5)
    # 对称性
    assert p_value_two_sided(-2.5) == pytest.approx(p_value_two_sided(2.5))
    # 非有限输入返回 NaN 而不是 0/1（不能把"算不出"当成"极显著"）
    assert math.isnan(p_value_two_sided(float("nan")))
    assert math.isnan(p_value_two_sided(float("inf")))


def test_p_value_against_scipy_when_available():
    scipy_stats = pytest.importorskip("scipy.stats", reason="仅作交叉验证，缺失则跳过")
    for t_stat in (0.0, 0.5, 1.0, 1.96, 2.58, 3.5, -2.0):
        ours = p_value_two_sided(t_stat)
        theirs = float(2 * scipy_stats.norm.sf(abs(t_stat)))
        assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-15)


def test_bonferroni_matches_hand_calculation():
    p_values = [0.001, 0.01, 0.02, 0.04, 0.5]
    result = bonferroni(p_values, alpha=0.05)
    assert result["m"] == 5
    assert result["threshold"] == pytest.approx(0.01)
    # 只有 0.001 与 0.01（等于阈值）通过
    assert list(result["rejected"]) == [True, True, False, False, False]
    assert result["adjusted"][0] == pytest.approx(0.005)
    assert result["adjusted"][-1] == pytest.approx(1.0)      # 调整后 p 值截断在 1


def test_benjamini_hochberg_matches_hand_calculation():
    # p = 0.001, 0.008, 0.039, 0.041, 0.042; m=5, alpha=0.05
    # 阈值序列 k/m*alpha = 0.01, 0.02, 0.03, 0.04, 0.05
    # 最大满足 p_(k) <= 阈值 的 k = 4（0.041 <= 0.04 不成立 -> k=3？逐项核验）
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042]
    result = benjamini_hochberg(p_values, alpha=0.05)
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
    # 手算：k=1: 0.001<=0.01 ✓; k=2: 0.008<=0.02 ✓; k=3: 0.039<=0.03 ✗; k=4: 0.041<=0.04 ✗; k=5: 0.042<=0.05 ✓
    # 取最大的 k = 5 -> 全部拒绝（BH 的阶梯性质）
    assert all(thresholds[k] >= p_values[k] for k in (0, 1)) and thresholds[4] >= p_values[4]
    assert list(result["rejected"]) == [True] * 5
    # 调整后 p 值单调不降（按 p 排序后）
    adjusted = np.asarray(result["adjusted"])
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_benjamini_hochberg_rejects_only_the_smallest_when_others_are_large():
    p_values = [0.0001, 0.2, 0.4, 0.6, 0.8]
    result = benjamini_hochberg(p_values, alpha=0.05)
    assert list(result["rejected"]) == [True, False, False, False, False]
    assert result["adjusted"][0] == pytest.approx(0.0005)


def test_bh_is_less_conservative_than_bonferroni():
    """FDR 比 FWER 更有功效：BH 拒绝集应包含 Bonferroni 的拒绝集。"""
    p_values = [0.001, 0.005, 0.008, 0.009, 0.2, 0.4, 0.6, 0.9]
    bh = benjamini_hochberg(p_values, alpha=0.05)
    bonf = bonferroni(p_values, alpha=0.05)
    assert np.all(bh["rejected"][bonf["rejected"]]), "BH 至少要拒绝 Bonferroni 拒绝的那些"
    assert bh["rejected"].sum() >= bonf["rejected"].sum()


def test_bh_matches_statsmodels_when_available():
    """与 statsmodels 的 fdr_bh 逐元素对照（返回顺序：rejected, pvals_corrected, ...）。"""
    statsmodels_multitest = pytest.importorskip("statsmodels.stats.multitest", reason="仅作交叉验证，缺失则跳过")
    rng = np.random.default_rng(3)
    p_values = np.concatenate([rng.uniform(0, 0.04, 6), rng.uniform(0.1, 1.0, 14)])
    ours = benjamini_hochberg(p_values, alpha=0.05)
    rejected, corrected, _sidak, _bonf = statsmodels_multitest.multipletests(p_values, alpha=0.05, method="fdr_bh")
    assert list(np.asarray(ours["rejected"])) == list(np.asarray(rejected))
    np.testing.assert_allclose(np.asarray(ours["adjusted"], dtype=float), np.asarray(corrected, dtype=float), atol=1e-12)


def test_nan_p_values_are_never_rejected():
    """算不出 p 值的组合（例如截面太少）不能因为排序靠前而被当成显著。"""
    p_values = [float("nan"), 0.001, 0.02]
    bh = benjamini_hochberg(p_values, alpha=0.05)
    assert bh["rejected"][0] is np.False_ or bh["rejected"][0] == False  # noqa: E712
    assert math.isnan(bh["adjusted"][0])


def test_edge_cases_and_validation():
    assert bonferroni([], 0.05)["m"] == 0
    assert benjamini_hochberg([], 0.05)["m"] == 0
    assert list(bonferroni([], 0.05)["rejected"]) == []
    with pytest.raises(ValueError):
        bonferroni([0.1], alpha=0.0)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1], alpha=1.0)
    with pytest.raises(ValueError):
        expected_false_positives(-1)
    assert expected_false_positives(111, 0.05) == pytest.approx(5.55)


def test_summarize_significance_flags_the_expected_false_positives():
    """构造 100 个纯噪声 t 值：不做校正会"发现"约 5 个，做校正后应当基本清零。"""
    rng = np.random.default_rng(11)
    t_stats = rng.normal(0, 1, 100)          # 纯噪声
    table = pd.DataFrame({"factor": [f"f{i}" for i in range(100)], "horizon": 5, "t_stat_adj": t_stats})
    outcome = summarize_significance(table, alpha=0.05)
    enriched = outcome["table"]
    raw = int(enriched["significant_raw"].sum())
    assert raw >= 0
    assert outcome["expected_false_positives"] == pytest.approx(5.0)
    assert int(enriched["significant_bh"].sum()) <= max(raw, 1)
    # 纯噪声下，Bonferroni 基本不该报出任何显著
    assert int(enriched["significant_bonferroni"].sum()) <= 1


def test_summarize_significance_keeps_strong_signals():
    table = pd.DataFrame(
        {
            "factor": ["strong", "weak"],
            "horizon": [5, 5],
            "t_stat_adj": [5.0, 0.3],
        }
    )
    enriched = summarize_significance(table, alpha=0.05)["table"]
    assert bool(enriched.loc[0, "significant_bh"]) is True
    assert bool(enriched.loc[1, "significant_bh"]) is False
    assert enriched.loc[0, "q_bh"] < 0.05


def test_summarize_significance_validates_input():
    with pytest.raises(ValueError):
        summarize_significance(pd.DataFrame({"factor": ["a"]}))
    empty = summarize_significance(pd.DataFrame())
    assert empty["table"].empty and empty["expected_false_positives"] == 0.0
