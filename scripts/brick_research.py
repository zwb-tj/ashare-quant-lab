"""砖型图 / B3 研究数字的复现脚本。

用途
----
``docs/research/brick-short-term.md``（第 10 节）与 ``docs/research/b3-dynamic-group.md``
引用了大量实测数字（价格 vs 砖色一致率 76.02%、B3 特征分布、各候选语句覆盖率/准确率）。
这些数字原先由临时脚本产生、随后丢失，导致文档**声称可复现但实际不可复现**。

本脚本用**本仓库自己的实现**（``brick_chart`` / ``B3Confirm``）在本地全市场日线缓存上
重算全部数字，并把结果写成：

* ``output/brick_research/*.csv``            —— 明细
* ``docs/research/brick-research-results.json`` —— 机器可读的数字清单（很小，进仓库）

清单是文档数字的**唯一事实来源**：``tests/test_research_docs.py`` 断言文档里出现的
每个数字都能在清单里找到。这样文档数字一旦漂移就会被测试抓住。

复现
----
    python scripts/brick_research.py                       # 用默认样本
    python scripts/brick_research.py --brick-limit 0       # 用全市场

样本口径固定为「按文件名排序取前 N 只」，因此结果**逐位可复现**。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqlab.indicators_extra import brick_chart  # noqa: E402
from aqlab.rules_zgnb import B3Confirm  # noqa: E402
from aqlab.study_universe import load_universe_daily  # noqa: E402

DAILY_DIR = ROOT / "data" / "universe" / "daily"
OUT_DIR = ROOT / "output" / "brick_research"
MANIFEST = ROOT / "docs" / "research" / "brick-research-results.json"

# 文档引用的样本口径（固定，勿随意改）
BRICK_LIMIT = 400
B3_LIMIT = 500


# --------------------------------------------------------------------------------------
# 第 10.1 节：价格方向 vs 砖色
# --------------------------------------------------------------------------------------
def price_vs_brick(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    """混淆矩阵：单日涨跌方向 vs 砖色（红/绿）。"""
    rows: list[pd.DataFrame] = []
    for _symbol, frame in frames.items():
        chart = brick_chart(frame.sort_index())
        close = frame["close"].astype(float).sort_index()
        ret = close.pct_change()
        red = chart["red"].astype(bool)
        green = chart["green"].astype(bool)
        rows.append(pd.DataFrame({"ret": ret, "red": red, "green": green}))
    all_rows = pd.concat(rows, ignore_index=True)

    valid = all_rows["ret"].notna() & (all_rows["red"] | all_rows["green"])
    d = all_rows.loc[valid]
    up = d["ret"] > 0.0
    down = d["ret"] < 0.0
    flat = d["ret"] == 0.0

    up_red = int((up & d["red"]).sum())
    down_green = int((down & d["green"]).sum())
    up_green = int((up & d["green"]).sum())
    down_red = int((down & d["red"]).sum())
    n_flat = int(flat.sum())
    total = len(d)

    agree = up_red + down_green
    return {
        "observations": total,
        "symbols": len(frames),
        "up_and_red": up_red,
        "down_and_green": down_green,
        "up_but_green": up_green,
        "down_but_red": down_red,
        "flat": n_flat,
        "up_and_red_pct": up_red / total * 100.0,
        "down_and_green_pct": down_green / total * 100.0,
        "up_but_green_pct": up_green / total * 100.0,
        "down_but_red_pct": down_red / total * 100.0,
        "flat_pct": n_flat / total * 100.0,
        "agreement_pct": agree / total * 100.0,
        "disagreement_pct": (up_green + down_red) / total * 100.0,
    }


# --------------------------------------------------------------------------------------
# B3：特征分布 + 候选语句覆盖率/准确率
# --------------------------------------------------------------------------------------
def _b3_features(frame: pd.DataFrame) -> pd.DataFrame:
    """逐日计算 B3 信号与各候选条件（全部只用当日及之前的信息）。"""
    df = frame.sort_index()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    prev_close = close.shift(1)

    sig = B3Confirm().signal(df).astype(bool)

    flat_open = (open_ / prev_close - 1.0).abs() <= 0.01
    small_body = ((close - open_).abs() / prev_close) <= 0.02

    # 近 3 日内出现过 >=4% 的大涨（shift(1)：不含当日）
    surge = (close / prev_close - 1.0 >= 0.04).rolling(3, min_periods=1).max().shift(1).fillna(False)
    ma20 = close.rolling(20, min_periods=20).mean()
    above_ma20 = close > ma20
    vol_ma10 = df["volume"].astype(float).rolling(10, min_periods=10).mean()
    heavy = df["volume"].astype(float) > vol_ma10
    light = df["volume"].astype(float) < vol_ma10

    # 用户原句的骨架：**前 3 日**（t-1/t-2/t-3）连阴 + **今日**收阳。
    # 注意不能写成 `down & down.shift(1) & down.shift(2)`——那把当日也算进"收阴"，
    # 而 B3 当日本身可能是小阴线，会把这个交集算错。
    down = (close < open_).fillna(False).astype(bool)
    prior3_down = (down.shift(1) & down.shift(2) & down.shift(3)).fillna(False).astype(bool)
    today_up = (close > open_).fillna(False).astype(bool)
    reversal_sentence = prior3_down & today_up          # 反包语义
    flat_open_or_up = flat_open | today_up

    # 前期超跌：两种都算，因为口径不同、数字不同（文档必须写明用的是哪种）
    #  E1 = 20 日累计跌幅超过 15%（收益口径）
    #  E2 = 价格比前 20 日最高点低 15% 以上（回撤口径）
    prior_dd_ret = (close / close.shift(20) - 1.0).shift(1) < -0.15
    prior_dd_high = (close / close.rolling(20, min_periods=20).max() - 1.0).shift(1) < -0.15

    return pd.DataFrame(
        {
            "signal": sig,
            "flat_open": flat_open,
            "small_body": small_body,
            "surge_3d": surge.astype(bool),
            "above_ma20": above_ma20.fillna(False),
            "heavy_vol": heavy.fillna(False),
            "light_vol": light.fillna(False),
            "prior3_down": prior3_down,
            "today_up": today_up,
            "reversal_sentence": reversal_sentence,
            "flat_or_up": flat_open_or_up,
            "prior_dd_ret": prior_dd_ret.fillna(False),
            "prior_dd_high": prior_dd_high.fillna(False),
        }
    )


def b3_study(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    allf = pd.concat([_b3_features(f) for f in frames.values()], ignore_index=True)

    sig = allf["signal"]
    n_sig = int(sig.sum())
    s = allf.loc[sig]

    # 用户原句骨架（前 3 日连阴 + 今日收阳）与 B3 的交集：用于证明两者语义不相交。
    reversal = allf["reversal_sentence"]
    n_rev = int(reversal.sum())
    overlap = int((reversal & sig).sum())

    features = pd.DataFrame(
        [
            ("平开 |open/prev_close-1| <= 1%", int(s["flat_open"].sum()), n_sig),
            ("小实体 |close-open|/prev_close <= 2%", int(s["small_body"].sum()), n_sig),
            ("近 3 日内有过 >=4% 大涨", int(s["surge_3d"].sum()), n_sig),
            ("股价在 20 日线上", int(s["above_ma20"].sum()), n_sig),
            ("量在 10 日线上", int(s["heavy_vol"].sum()), n_sig),
            ("前 3 日（t-1..t-3）连阴", int(s["prior3_down"].sum()), n_sig),
        ],
        columns=["feature", "hits", "n_signals"],
    )
    features["pct"] = features["hits"] / features["n_signals"] * 100.0

    shape = allf["flat_open"] & allf["small_body"]
    b = shape & allf["surge_3d"]
    ma = allf["above_ma20"]
    heavy = allf["heavy_vol"]
    light = allf["light_vol"]
    dd_ret = allf["prior_dd_ret"]
    dd_high = allf["prior_dd_high"]

    cands = {
        "A 平开 + 小实体": shape,
        "B A + 近 3 日大涨 4%": b,
        "C B + 20 日线上": b & ma,
        "D B + 量在 10 日线上": b & heavy,
        "E B + 前期超跌（20 日累计跌幅 >15%）": b & dd_ret,
        "E2 B + 前期超跌（距 20 日高点回撤 >15%）": b & dd_high,
        "F E + 20 日线上": b & dd_ret & ma,
        "G B + 缩量（量 < 10 日均量）": b & light,
        "H B + 20 日线上 + 缩量": b & ma & light,
    }
    rows = []
    for label, mask in cands.items():
        selected = int(mask.sum())
        hit = int((mask & sig).sum())
        rows.append(
            {
                "sentence": label,
                "selected": selected,
                "hits": hit,
                "coverage_pct": hit / n_sig * 100.0 if n_sig else 0.0,
                "precision_pct": hit / selected * 100.0 if selected else 0.0,
            }
        )
    sentences = pd.DataFrame(rows)
    base = sentences.loc[0, "precision_pct"]
    sentences["lift"] = np.where(base > 0, sentences["precision_pct"] / base, 0.0)

    extra = {
        "n_symbols": len(frames),
        "n_signals": n_sig,
        "reversal_sentence_selected": n_rev,
        "reversal_sentence_overlap_with_b3": overlap,
        "b3_with_prior3_down": int(s["prior3_down"].sum()),
    }
    return features, sentences, extra


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="复现砖型图 / B3 研究数字")
    ap.add_argument("--daily-dir", default=str(DAILY_DIR))
    ap.add_argument("--brick-limit", type=int, default=BRICK_LIMIT, help="0 = 全市场")
    ap.add_argument("--b3-limit", type=int, default=B3_LIMIT, help="0 = 全市场")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    daily = Path(args.daily_dir)
    if not daily.is_dir():
        print(f"找不到日线目录：{daily}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    brick_limit = args.brick_limit or None
    b3_limit = args.b3_limit or None

    print("[1/2] 价格方向 vs 砖色 ...")
    t0 = time.time()
    brick_frames = load_universe_daily(daily, limit=brick_limit)
    pvb = price_vs_brick(brick_frames)
    print(f"      {pvb['symbols']} 只 / {pvb['observations']} 个观测 / {time.time() - t0:.1f}s")

    print("[2/2] B3 特征分布与候选语句 ...")
    t0 = time.time()
    b3_frames = load_universe_daily(daily, limit=b3_limit)
    features, sentences, extra = b3_study(b3_frames)
    print(f"      {len(b3_frames)} 只 / {time.time() - t0:.1f}s")

    # 明细（output/ 被 gitignore，属临时产物）
    pd.DataFrame([pvb]).to_csv(out_dir / "price_vs_brick.csv", index=False)
    features.to_csv(out_dir / "b3_features.csv", index=False)
    sentences.to_csv(out_dir / "b3_sentences.csv", index=False)

    manifest = {
        "_note": "由 scripts/brick_research.py 生成；docs/research 的数字以此为准。",
        "price_vs_brick": {k: round(v, 2) if isinstance(v, float) else v for k, v in pvb.items()},
        "b3_features": {
            row["feature"]: {"hits": int(row["hits"]), "pct": round(float(row["pct"]), 1)}
            for _i, row in features.iterrows()
        },
        "b3_sentences": {
            row["sentence"]: {
                "selected": int(row["selected"]),
                "coverage_pct": round(float(row["coverage_pct"]), 1),
                "precision_pct": round(float(row["precision_pct"]), 2),
                "lift": round(float(row["lift"]), 2),
            }
            for _i, row in sentences.iterrows()
        },
        "b3_extra": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in extra.items()},
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"一致率 {pvb['agreement_pct']:.2f}% / 不一致率 {pvb['disagreement_pct']:.2f}%")
    print(features.to_string(index=False))
    print()
    print(sentences.to_string(index=False))
    print(f"\n明细 → {out_dir}\n清单 → {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
