"""Round-robin several tunings of our own bot and rank them by seat-A wins.

Why seat A only. A mirror match -- byte-identical `MM_TUNE` on both sides -- does not draw. It
returns a clean 1W1L with mirrored statistics, and seat B is always the winner: within a tick
the engine collects both fleets' orders, but seat B's strategy runs against a state that
already reflects seat A's move, so B is effectively half a tick ahead. That bias is worth more
than most tuning differences, which makes a 2-game head-to-head unable to resolve anything
small -- it reports 1W1L for "no difference" and for "slightly better" alike.

Holding the seat fixed removes the bias instead of averaging over it. Every variant plays every
other **from seat A**, where winning requires actually being stronger. A variant that wins k of
its n seat-A games beat the bias k times, and that count is comparable across variants.

    python3 tools/tournament.py "" "PAYLOAD_ANCHOR_COUNT=4" "PAYLOAD_ANCHOR_COUNT=8,SPACING_WEIGHT=0.8"

Also reports each variant's seat-A results against the fixed opponent panel, if one is built,
since beating an outside bot is the thing the tournament actually scores.
"""

import argparse
import concurrent.futures
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from headtohead import duel, make_runner  # noqa: E402

BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+", help='MM_TUNE strings; "" is the defaults')
    parser.add_argument("--panel", default="/tmp/mm-panel")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    root = tempfile.mkdtemp(prefix="mm-rr-")
    runners = {
        tune: make_runner(root, f"v{i}", tune) for i, tune in enumerate(args.variants)
    }

    panel: List[Tuple[str, str]] = []
    if os.path.isdir(args.panel):
        for name in sorted(os.listdir(args.panel)):
            run = os.path.join(args.panel, name, "run")
            if os.path.exists(run):
                panel.append((name, run))

    # Every job is a seat-A game for the variant named first.
    jobs: List[Tuple[str, str, str, str]] = []
    for tune, run in runners.items():
        for other, other_run in runners.items():
            if tune != other:
                jobs.append((tune, "rr", run, other_run))
        for name, run_p in panel:
            jobs.append((tune, f"panel:{name}", run, run_p))

    results: Dict[str, Dict[str, int]] = {
        tune: {"rr_win": 0, "rr_games": 0, "panel_win": 0, "panel_games": 0, "ticks": 0}
        for tune in runners
    }

    def play_one(job):
        tune, kind, a, b = job
        log = os.path.join(root, f"{abs(hash(job))}.mmgl")
        winner, summary = duel(a, b, log, args.timeout)
        return tune, kind, winner, summary

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for tune, kind, winner, summary in pool.map(play_one, jobs):
            row = results[tune]
            won = winner == "A"
            if kind == "rr":
                row["rr_games"] += 1
                row["rr_win"] += 1 if won else 0
            else:
                row["panel_games"] += 1
                row["panel_win"] += 1 if won else 0
            row["ticks"] += summary.get("ticks", 0) or 0

    print(f"{'variant':<56} {'seatA vs peers':>15} {'seatA vs panel':>15} {'avg ticks':>10}")
    ranked = sorted(
        results.items(),
        key=lambda kv: (
            -(kv[1]["rr_win"] + kv[1]["panel_win"]),
            kv[1]["ticks"],
        ),
    )
    for tune, row in ranked:
        games = max(1, row["rr_games"] + row["panel_games"])
        label = tune if tune else "(defaults)"
        print(f"{label:<56} {row['rr_win']:>7}/{row['rr_games']:<7} "
              f"{row['panel_win']:>7}/{row['panel_games']:<7} "
              f"{row['ticks'] / games:>10.0f}")

    print("\nseat A is the discriminating seat: seat B wins mirror matches for free, so a")
    print("seat-A win means the variant was genuinely stronger, not better seated.")

    for entry in os.listdir(root):
        try:
            os.remove(os.path.join(root, entry))
        except OSError:
            pass
    try:
        os.rmdir(root)
    except OSError:
        pass


if __name__ == "__main__":
    main()
