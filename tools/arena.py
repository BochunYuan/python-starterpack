"""Run a batch of matches between two bots and report the aggregate, not one match's luck.

Why this exists: `eval_tick` shuffles `bot_actions` every tick, and `closer` breaks an exact
distance tie with a coin flip. Two *identical* strategies therefore do not draw -- whichever
side wins the first fight has the larger fleet, which wins the next one, and a single
self-play match ends 3-vs-29 on noise alone. One match tells you almost nothing; twenty tell
you whether a change helped.

Each pairing is played twice per seed -- once with each bot as team A -- because the sides are
not symmetric in practice even though the engine mirrors the world: team A's orders are
generated first, and the spawn corner differs. Swapping and summing removes that.

    python3 tools/arena.py --a .mm/run --b /tmp/baseline-bot/run --games 10

Prints a win/loss/draw record from team A's point of view, plus the averages that explain it.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, summarize  # noqa: E402


ENGINE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mm", "bin", "mm-engine"
)


def play(bot_a: str, bot_b: str, log_path: str, timeout: int) -> Optional[dict]:
    """One match. Returns the parsed summary, or None if the engine failed to produce a log."""
    cmd = [ENGINE, bot_a, bot_b, "-o", f"g:{log_path}"]
    try:
        proc = subprocess.run(
            cmd, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=False, text=True,
        )
    except subprocess.TimeoutExpired:
        return None
    if not os.path.exists(log_path):
        return None
    config, states, winner = load(log_path)
    if not states:
        return None
    summary = summarize(config, states, winner)
    # The result line goes to the engine's stdout, not into the `-o g:` gamelog, so the log
    # alone cannot say who won -- `load` reports None for every match. Read it off stdout and
    # let it override.
    if summary.get("winner") is None:
        summary["winner"] = _winner_from_stdout(proc.stdout or "")
    return summary


def _winner_from_stdout(text: str) -> Optional[str]:
    """`A`, `B`, or None for a draw, from the engine's `{"winner": ...}` line."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "winner" in row:
            return row["winner"]
    return None


def _score(summary: dict, ours_is_a: bool) -> str:
    """Win, loss or draw from *our* point of view, whichever side we played.

    The engine writes `{"winner": "A"}`, `{"winner": "B"}`, or no winner line at all for a
    draw. `summarize` normalises a missing line to None, which is also what an unfinished log
    looks like -- but a log with `final_tick == max_ticks` and no winner is a genuine draw on
    the tiebreak chain, so the two cases are distinguished by the tick count upstream.
    """
    winner = summary.get("winner")
    if winner is None:
        return "draw"
    we_won = (winner == "A") if ours_is_a else (winner == "B")
    return "win" if we_won else "loss"


def _ours(summary: dict, ours_is_a: bool, key: str):
    """Read a per-team stat for our side, given which side we played."""
    suffix = "a" if ours_is_a else "b"
    return summary.get(f"{key}_{suffix}")


def run_pairing(
    ours: str, theirs: str, tmp: str, tag: str, timeout: int, jobs: int
) -> Tuple[Dict[str, int], Dict[str, List[float]]]:
    """Play `ours` against `theirs` in both seats and aggregate the result.

    Both seats and not more: the pairing is deterministic in practice, so a third match adds
    a duplicate row rather than a sample. Variety comes from the opponent list, not repetition.
    """
    pairings = [
        (ours, theirs, os.path.join(tmp, f"{tag}-a.mmgl"), True),
        (theirs, ours, os.path.join(tmp, f"{tag}-b.mmgl"), False),
    ]

    record: Dict[str, int] = {"win": 0, "loss": 0, "draw": 0, "error": 0}
    stats: Dict[str, List[float]] = {}

    def track(name: str, value) -> None:
        if isinstance(value, (int, float)):
            stats.setdefault(name, []).append(float(value))

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(play, a, b, path, timeout): ours_a
            for a, b, path, ours_a in pairings
        }
        for done in concurrent.futures.as_completed(futures):
            ours_a = futures[done]
            summary = done.result()
            if summary is None:
                record["error"] += 1
                continue
            record[_score(summary, ours_a)] += 1
            # `capture` is signed toward team B's goal, so from the B seat the sign flips.
            cap = summary.get("final_capture") or 0.0
            track("final_capture", cap if ours_a else -cap)
            for key in ("health", "tokens", "deaths", "avg_slots", "fleet"):
                track(key, _ours(summary, ours_a, key))
            track("ticks", summary.get("ticks"))

    for _, _, path, _ in pairings:
        try:
            os.remove(path)
        except OSError:
            pass
    return record, stats


def _mean(values: Optional[List[float]]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="bot under test (run script)")
    parser.add_argument("--b", help="single opponent (run script)")
    parser.add_argument("--panel", help="directory of opponents, each <dir>/<name>/run")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--label", default="", help="prefix for the summary line")
    args = parser.parse_args()

    opponents: List[Tuple[str, str]] = []
    if args.panel:
        for name in sorted(os.listdir(args.panel)):
            run = os.path.join(args.panel, name, "run")
            if os.path.exists(run):
                opponents.append((name, run))
    if args.b:
        opponents.append((os.path.basename(os.path.dirname(args.b)) or "opponent", args.b))
    if not opponents:
        print("no opponents: pass --b and/or --panel")
        return

    tmp = tempfile.mkdtemp(prefix="mm-arena-")
    total = {"win": 0, "loss": 0, "draw": 0, "error": 0}
    all_stats: Dict[str, List[float]] = {}

    print(f"{'opponent':<12} {'result':<7} {'capture':>8} {'health':>8} {'fleet':>6} "
          f"{'deaths':>7} {'slots':>6} {'ticks':>7}")
    for name, run in opponents:
        record, stats = run_pairing(args.a, run, tmp, name, args.timeout, args.jobs)
        for key in total:
            total[key] += record[key]
        for key, values in stats.items():
            all_stats.setdefault(key, []).extend(values)
        result = f"{record['win']}W{record['loss']}L{record['draw']}D"
        print(f"{name:<12} {result:<7} {_mean(stats.get('final_capture')):8.3f} "
              f"{_mean(stats.get('health')):8.1f} {_mean(stats.get('fleet')):6.1f} "
              f"{_mean(stats.get('deaths')):7.1f} {_mean(stats.get('avg_slots')):6.2f} "
              f"{_mean(stats.get('ticks')):7.0f}")

    played = total["win"] + total["loss"] + total["draw"]
    points = total["win"] * 3 + total["draw"]
    prefix = f"{args.label}: " if args.label else ""
    print(f"\n{prefix}{total['win']}W {total['loss']}L {total['draw']}D "
          f"errors={total['error']} of {played}")
    if played:
        print(f"{prefix}points {points}/{played * 3} ({points / (played * 3):.1%})  "
              f"avg capture {_mean(all_stats.get('final_capture')):+.3f}  "
              f"avg health {_mean(all_stats.get('health')):.1f}")

    try:
        os.rmdir(tmp)
    except OSError:
        pass


if __name__ == "__main__":
    main()
