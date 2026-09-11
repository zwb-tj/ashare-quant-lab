"""Named rule profiles: switch the daily pipeline between rule sets with one flag.

A profile is just a list of ``(rule_name, params, weight)`` bindings, so the
pipeline, the agent tools and the CLI all stay unaware of the strategy names.

* ``generic``        — the three generic patterns (pullback / needle shadow / volume-price surge)
* ``b1`` / ``b2`` / ``b3``            — single-signal profiles
* ``needle_20`` / ``needle_30``       — 单针下20 / 单针下30（RSL 口径）
* ``volume_price_v3``                 — 量价齐升 V3
* ``zgnb_full``      — B1 + B2 + B3 + 单针下20 + 量价齐升V3 的组合评分
"""

from __future__ import annotations

from typing import Any

from aqlab.rules import DEFAULT_RULE_BINDINGS, ActivityValueGate
from aqlab.rules_zgnb import ActiveMarketValueGate

__all__ = ["PROFILES", "load_profile", "list_profiles", "build_gate", "GATES"]

PROFILES: dict[str, list[tuple[str, dict[str, Any], float]]] = {
    "generic": list(DEFAULT_RULE_BINDINGS),
    "b1": [("b1_opportunity", {}, 1.0)],
    "b2": [("b2_confirm", {}, 1.0)],
    "b3": [("b3_confirm", {}, 1.0)],
    "needle_20": [("needle_rsl", {"short_max": 20.0, "long_min": 80.0}, 1.0)],
    "needle_30": [("needle_rsl", {"short_max": 30.0, "long_min": 85.0, "long_strict": True}, 1.0)],
    "volume_price_v3": [("volume_price_v3", {}, 1.0)],
    "zgnb_full": [
        ("b1_opportunity", {}, 0.25),
        ("b2_confirm", {}, 0.20),
        ("b3_confirm", {}, 0.15),
        ("needle_rsl", {"short_max": 20.0, "long_min": 80.0}, 0.15),
        ("volume_price_v3", {}, 0.25),
    ],
    "zgnb_needle30": [
        ("needle_rsl", {"short_max": 30.0, "long_min": 85.0, "long_strict": True}, 0.6),
        ("volume_price_v3", {}, 0.4),
    ],
}

GATES = {
    "amv": ActiveMarketValueGate,        # 0AMV 活跃市值 + 波段开关（个人口径）
    "activity": ActivityValueGate,       # 通用活跃度开关（收盘价×成交量双均线）
    "none": None,
}


def load_profile(name: str) -> list[tuple[str, dict[str, Any], float]]:
    """Return a fresh copy of the bindings for ``name``."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile '{name}'; available: {sorted(PROFILES)}")
    return [(rule, dict(params), weight) for rule, params, weight in PROFILES[name]]


def list_profiles() -> dict[str, list[str]]:
    """Summarise the profiles for the CLI / docs."""
    return {name: [rule for rule, _params, _weight in bindings] for name, bindings in PROFILES.items()}


def build_gate(name: str):
    """Instantiate a market gate by name (``amv`` / ``activity`` / ``none``)."""
    if name not in GATES:
        raise KeyError(f"unknown gate '{name}'; available: {sorted(GATES)}")
    factory = GATES[name]
    return None if factory is None else factory()
