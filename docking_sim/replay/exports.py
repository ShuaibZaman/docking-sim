"""Shareable replay bundles for exact benchmark missions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from docking_sim.replay.benchmarks import BENCHMARK_PROTOCOL_VERSION


EXPORT_KIND = "docking-lab-replay"


def export_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap an exact-scenario replay so it can be saved or shared as JSON."""

    scenario = dict(payload.get("scenario") or {})
    panels = []
    for panel in payload.get("panels") or []:
        replay = dict(panel.get("replay") or {})
        if replay.get("scenario_id") and scenario.get("id") and replay["scenario_id"] != scenario["id"]:
            raise ValueError("Export refused: a panel replayed a different scenario.")
        if (
            replay.get("scenario_hash")
            and scenario.get("scenario_hash")
            and replay["scenario_hash"] != scenario["scenario_hash"]
        ):
            raise ValueError("Export refused: a panel replayed a different mission hash.")
        panels.append(
            {
                "candidate": panel.get("candidate"),
                "label": panel.get("label"),
                "replay": replay,
            }
        )
    return {
        "kind": EXPORT_KIND,
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": payload.get("benchmark"),
        "scenario": scenario,
        "panels": panels,
    }
