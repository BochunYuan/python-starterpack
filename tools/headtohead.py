"""Play two tunings of our own bot against each other, directly.

The opponent panel has stopped discriminating -- every variant beats all five, so the record
is 10W0L0D whatever we change and only the margins move. Margins against weak opponents are a
weak signal: they say how fast we beat a bot that was going to lose anyway.

A challenger-versus-champion match is the direct question instead: does configuration A beat
configuration B, head to head, when both sides play well? That is also much closer to the
tournament, where every opponent is someone's real submission.

    python3 tools/headtohead.py "PAYLOAD_ANCHOR_COUNT=14" ""
    python3 tools/headtohead.py "PAYLOAD_ANCHOR_COUNT=14,SPACING_WEIGHT=0.8" "PAYLOAD_ANCHOR_COUNT=2"

Each side gets a wrapper script that bakes its own `MM_TUNE` in, so the two configurations
differ only in that variable. Both seats are played, since seat A acts first.
"""

import os
import stat
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, summarize  # noqa: E402
from arena import _winner_from_stdout  # noqa: E402

BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(BOT, ".mm", "bin", "mm-engine")


def make_runner(root: str, name: str, tune: str) -> str:
    """A run script for our bot with `MM_TUNE` fixed to `tune`."""
    path = os.path.join(root, f"run-{name}")
    with open(path, "w") as handle:
        handle.write("#!/bin/sh\n")
        # Set rather than exported-from-parent, so the two sides cannot pick up each other's
        # value from the engine's environment.
        handle.write(f'MM_TUNE="{tune}"\nexport MM_TUNE\n')
        handle.write(f'exec python3 "{BOT}/__main__.py" "$1"\n')
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def duel(run_a: str, run_b: str, log: str, timeout: int = 900) -> Tuple[Optional[str], dict]:
    """One match. Returns (winner, summary) with the winner as reported by the engine."""
    proc = subprocess.run(
        [ENGINE, run_a, run_b, "-o", f"g:{log}"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        timeout=timeout, check=False,
    )
    winner = _winner_from_stdout(proc.stdout or "")
    summary = {}
    if os.path.exists(log):
        config, states, _ = load(log)
        if states:
            summary = summarize(config, states, winner)
        os.remove(log)
    return winner, summary


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: headtohead.py "<challenger tune>" ["<champion tune>"]')
        return
    challenger = sys.argv[1]
    champion = sys.argv[2] if len(sys.argv) > 2 else ""

    root = tempfile.mkdtemp(prefix="mm-h2h-")
    run_c = make_runner(root, "challenger", challenger)
    run_d = make_runner(root, "champion", champion)

    print(f"challenger: {challenger or '(defaults)'}")
    print(f"champion:   {champion or '(defaults)'}\n")

    wins = losses = draws = 0
    for seat, (a, b, challenger_is_a) in enumerate((
        (run_c, run_d, True),
        (run_d, run_c, False),
    )):
        winner, summary = duel(a, b, os.path.join(root, f"h2h{seat}.mmgl"))
        if winner is None:
            outcome = "draw"
            draws += 1
        else:
            won = (winner == "A") == challenger_is_a
            outcome = "challenger WINS" if won else "champion wins"
            if won:
                wins += 1
            else:
                losses += 1
        cap = summary.get("final_capture", 0.0) or 0.0
        # Report capture from the challenger's point of view.
        cap = cap if challenger_is_a else -cap
        seat_name = "challenger as A" if challenger_is_a else "challenger as B"
        print(f"{seat_name:<18} {outcome:<16} ticks={summary.get('ticks', 0):>5} "
              f"capture={cap:+.3f} "
              f"fleet={summary.get('fleet_a' if challenger_is_a else 'fleet_b', 0):>3}v"
              f"{summary.get('fleet_b' if challenger_is_a else 'fleet_a', 0):<3} "
              f"health={summary.get('health_a' if challenger_is_a else 'health_b', 0):>6.1f}v"
              f"{summary.get('health_b' if challenger_is_a else 'health_a', 0):<6.1f}")

    print(f"\nchallenger record: {wins}W {losses}L {draws}D of 2")
    # Calibrated against a mirror match -- identical `MM_TUNE` on both sides -- which does NOT
    # draw: it returns 1W1L with byte-identical mirrored statistics, because seat B acts on
    # state that already reflects seat A's orders for that tick. So a 1W1L is exactly what two
    # equally strong configurations produce, and reading it as a regression is reading noise.
    # Only a sweep of both seats is evidence, since that means the challenger overcame the seat
    # advantage in the seat where it did not have it.
    if wins == 2:
        print("=> challenger wins from both seats, beating the seat bias. Real improvement.")
    elif losses == 2:
        print("=> champion wins from both seats. Reject the change.")
    else:
        print("=> 1W1L is the mirror-match baseline: no measurable difference. "
              "Seat, not strategy, decided these.")

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
