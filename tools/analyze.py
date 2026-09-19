"""Reconstruct a match from a .mmgl gamelog and report what actually happened.

The log is a diff stream: line 1 is the whole `GameConfig`, line 2 is the first full
`GameState`, and every line after that is a nested diff against the state before it. Lines
beginning with `#` are comments the engine writes (elapsed time, and a marker when time
enforcement was disabled), and the result line is a bare `{"winner": ...}`.

Run it on a log:

    python3 tools/analyze.py /tmp/m1.mmgl

It exists because "did that change help?" is not answerable from the winner line alone --
a self-play match between two copies of one strategy is decided by the tiebreak chain, so
the interesting numbers are the payload track, the health pool and the token curve.
"""

import json
import sys
from typing import Any, Dict, List, Optional, Tuple


def _merge(base: Any, patch: Any) -> Any:
    """Apply one gamelog diff.

    Fleets use `{"added": {...}, "changed": {...}, "removed": [...]}`; everything else is a
    plain nested dict merge. `added` carries a whole bot, `changed` only the fields that
    moved, so a changed entry merges into the bot already there.
    """
    if not isinstance(patch, dict):
        return patch
    if not isinstance(base, dict):
        base = {}

    if "added" in patch or "changed" in patch or "removed" in patch:
        out = dict(base)
        for bot_id in patch.get("removed", ()):
            out.pop(str(bot_id), None)
        for bot_id, bot in (patch.get("added") or {}).items():
            out[str(bot_id)] = bot
        for bot_id, fields in (patch.get("changed") or {}).items():
            cur = dict(out.get(str(bot_id)) or {})
            cur.update(fields)
            out[str(bot_id)] = cur
        return out

    out = dict(base)
    for key, value in patch.items():
        out[key] = _merge(out.get(key), value)
    return out


def _as_fleet(value: Any) -> Dict[str, dict]:
    """Normalise a fleet to id -> bot. The first full state writes a list, diffs a dict."""
    if isinstance(value, list):
        return {str(bot["id"]): bot for bot in value}
    return dict(value or {})


def load(path: str) -> Tuple[dict, List[dict], Optional[str]]:
    """The config, every reconstructed tick, and the winner if the log records one."""
    config: Optional[dict] = None
    winner: Optional[str] = None
    states: List[dict] = []
    cur: Optional[dict] = None

    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if config is None:
                config = row
                continue
            if "winner" in row and "tick" not in row:
                winner = row["winner"]
                continue
            if cur is None:
                cur = dict(row)
                # The first full state serialises each fleet as a *list* of bots, while every
                # diff after it is keyed by id. Normalising here rather than per snapshot is
                # load-bearing: merging an `{"added"/"changed"}` patch onto a list would fall
                # through `_merge`'s dict check and silently discard the whole fleet.
                cur["fleet_a"] = _as_fleet(cur.get("fleet_a"))
                cur["fleet_b"] = _as_fleet(cur.get("fleet_b"))
            else:
                cur = _merge(cur, row)
            snap = dict(cur)
            snap["fleet_a"] = dict(cur.get("fleet_a") or {})
            snap["fleet_b"] = dict(cur.get("fleet_b") or {})
            states.append(snap)

    return config or {}, states, winner


def _popcount(n: int) -> int:
    c = 0
    while n:
        n &= n - 1
        c += 1
    return c


def summarize(config: dict, states: List[dict], winner: Optional[str]) -> dict:
    """The numbers worth comparing between two versions of a strategy."""
    if not states:
        return {"error": "no states"}

    last = states[-1]
    classes = ("Battle", "Healer", "Extractor")

    def counts(fleet: Dict[str, dict]) -> Dict[str, int]:
        out = {c: 0 for c in classes}
        for bot in fleet.values():
            special = bot.get("special") or {}
            for name in classes:
                if name in special:
                    out[name] += 1
        return out

    def health(fleet: Dict[str, dict]) -> float:
        return sum(float(b.get("health", 0.0)) for b in fleet.values())

    # Extraction slot occupancy, averaged over the match: the economy's real throughput,
    # since a bot that holds no slot earns nothing however long it stands there.
    slot_ticks = {"a": 0, "b": 0}
    capture_peak = {"a": 0.0, "b": 0.0}
    deaths = {"a": 0, "b": 0}
    prev_ids = {"a": set(), "b": set()}
    contested = 0
    moving = 0

    for snap in states:
        for team in ("a", "b"):
            dep_mine = snap.get(f"deposit_a") or {}
            dep_theirs = snap.get(f"deposit_b") or {}
            for dep in (dep_mine, dep_theirs):
                ex = dep.get("extractors") or {}
                slot_ticks[team] += _popcount(int(ex.get(team, 0) or 0))
            ids = set((snap.get(f"fleet_{team}") or {}).keys())
            deaths[team] += len(prev_ids[team] - ids)
            prev_ids[team] = ids
        cap = float(snap.get("capture", 0.0))
        capture_peak["a"] = max(capture_peak["a"], cap)
        capture_peak["b"] = min(capture_peak["b"], cap)

    ticks = len(states)
    for i in range(1, ticks):
        if float(states[i].get("capture", 0.0)) != float(states[i - 1].get("capture", 0.0)):
            moving += 1
        else:
            contested += 1

    return {
        "winner": winner,
        "ticks": ticks,
        "final_tick": last.get("tick"),
        "final_capture": round(float(last.get("capture", 0.0)), 5),
        "capture_best_a": round(capture_peak["a"], 5),
        "capture_best_b": round(capture_peak["b"], 5),
        "ticks_payload_moving": moving,
        "ticks_payload_still": contested,
        "fleet_a": len(last.get("fleet_a") or {}),
        "fleet_b": len(last.get("fleet_b") or {}),
        "classes_a": counts(last.get("fleet_a") or {}),
        "classes_b": counts(last.get("fleet_b") or {}),
        "health_a": round(health(last.get("fleet_a") or {}), 2),
        "health_b": round(health(last.get("fleet_b") or {}), 2),
        "tokens_a": round(float((last.get("fabricator_a") or {}).get("tokens", 0.0)), 1),
        "tokens_b": round(float((last.get("fabricator_b") or {}).get("tokens", 0.0)), 1),
        "deaths_a": deaths["a"],
        "deaths_b": deaths["b"],
        "slot_ticks_a": slot_ticks["a"],
        "slot_ticks_b": slot_ticks["b"],
        "avg_slots_a": round(slot_ticks["a"] / max(1, ticks), 2),
        "avg_slots_b": round(slot_ticks["b"] / max(1, ticks), 2),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: analyze.py <log.mmgl> [more.mmgl ...]")
        return
    for path in sys.argv[1:]:
        config, states, winner = load(path)
        print(f"=== {path} ===")
        summary = summarize(config, states, winner)
        for key, value in summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
