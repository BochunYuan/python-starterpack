"""Measure one tuning knob at a time against the opponent panel.

The bot already beats every panel opponent, so the win/loss record has stopped discriminating
between variants. What still discriminates is *how* it wins:

- `ticks` -- how long the win took. A decisive strategy ends the match early; the clock
  running out means the tiebreak chain decided it, which is a much narrower win.
- `deaths` -- bodies lost getting there. Fewer deaths is a bigger surviving fleet, which is
  the second tiebreak and the thing that keeps the endgame elimination rule harmless.
- `capture` -- final payload progress, the first tiebreak.

So the score below is built from those, with the record as a hard gate: any variant that
actually loses a match is rejected no matter how good its averages look.

    python3 tools/ablate.py                      # test the built-in knob sweep
    python3 tools/ablate.py RALLY_COMMIT_FIGHTERS=1 RALLY_COMMIT_FIGHTERS=8

Each variant is run through the full panel in both seats. `MM_TUNE` reaches the bot because
the engine inherits this process's environment and the bot inherits the engine's.
"""

import argparse
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena import run_pairing, _mean  # noqa: E402


BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS = os.path.join(BOT, ".mm", "run")
PANEL = "/tmp/mm-panel"


# The knobs worth questioning, and the values to try. Each entry is one `MM_TUNE` string, so a
# variant can flip several knobs together where they only make sense as a pair.
SWEEP: List[str] = [
    "",                                   # the current defaults, as the reference row
    # Is the rally/wave mechanism earning its keep now that the fire bug is fixed? It was added
    # to stop reinforcements trickling in one at a time, but the trickle may have been a
    # symptom of bots never firing rather than of arriving alone.
    "RALLY_COMMIT_FIGHTERS=1",            # effectively disables gathering
    "RALLY_COMMIT_FIGHTERS=8",            # much larger waves
    # How many bodies hold the ring. Under the verified rule one is enough to push, so this is
    # purely about surviving the loss of an anchor.
    "PAYLOAD_ANCHOR_COUNT=1",
    "PAYLOAD_ANCHOR_COUNT=4",
    # Standoff distance. The fire bug was caused by holding a band with no sightline; the fix
    # closes when there is no shot, so the band itself may now be either safe or pointless.
    "ENGAGE_RANGE_FRACTION=0.5",
    "ENGAGE_RANGE_FRACTION=1.0",
    # Spacing against splash. Cheap to be wrong in either direction, worth knowing.
    "SPACING_WEIGHT=0.0",                 # no spacing at all
    "SPACING_WEIGHT=0.8",
    # Economy size.
    "EXTRACTOR_FLEET_CEILING=0.25",
    "EXTRACTOR_FLEET_CEILING=0.55",
    # The fighting floor that keeps the economy from eating the army.
    "FIGHTER_MATCH_SHARE=0.3",
    "FIGHTER_MATCH_SHARE=0.7",
    # Healers: are they worth their slot at all, given one healer exactly cancels one battle
    # bot's sustained damage?
    "COMPOSITION_HEALER=0.0",
    "COMPOSITION_HEALER=0.3",
    # The ambiguity itself, measured rather than reasoned about. Expected to be worse -- massing
    # buys no speed under the verified rule and feeds enemy splash -- and confirming that is
    # the point.
    "AMBIGUITY_PAYLOAD_SPEED_SCALES_WITH_BOT_COUNT=true",
]


def score(record: Dict[str, int], stats: Dict[str, List[float]], max_ticks: int = 9000) -> float:
    """A single comparable number, higher is better.

    Wins dominate, then speed of the win, then bodies kept. The weights are chosen so that
    no amount of tidy averages can outrank an extra win, and so a match that goes to the clock
    scores clearly below one that ends early.
    """
    played = record["win"] + record["loss"] + record["draw"]
    if not played:
        return -1e9
    points = record["win"] * 3 + record["draw"]
    # Fraction of the clock saved, averaged. A 3000-tick win beats a 9000-tick one.
    speed = 1.0 - (_mean(stats.get("ticks")) / max_ticks)
    deaths = _mean(stats.get("deaths"))
    return points * 100.0 + speed * 30.0 - deaths * 0.5


def run_variant(tune: str, jobs: int, timeout: int) -> Tuple[Dict[str, int], Dict[str, List[float]]]:
    """Play the whole panel with `MM_TUNE=tune` set for our bot."""
    if tune:
        os.environ["MM_TUNE"] = tune
    else:
        os.environ.pop("MM_TUNE", None)

    opponents = []
    for name in sorted(os.listdir(PANEL)):
        run = os.path.join(PANEL, name, "run")
        if os.path.exists(run):
            opponents.append((name, run))

    total = {"win": 0, "loss": 0, "draw": 0, "error": 0}
    merged: Dict[str, List[float]] = {}
    tmp = tempfile.mkdtemp(prefix="mm-ablate-")
    losses: List[str] = []
    try:
        for name, run in opponents:
            record, stats = run_pairing(OURS, run, tmp, name, timeout, jobs)
            for key in total:
                total[key] += record[key]
            for key, values in stats.items():
                merged.setdefault(key, []).extend(values)
            if record["loss"] or record["draw"]:
                losses.append(f"{name}({record['win']}W{record['loss']}L{record['draw']}D)")
    finally:
        try:
            os.rmdir(tmp)
        except OSError:
            pass
    merged["_losses"] = losses  # type: ignore[assignment]
    return total, merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", help="MM_TUNE strings; default is the sweep")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    variants = args.variants if args.variants else SWEEP
    if args.variants and "" not in variants:
        variants = [""] + variants

    print(f"{'variant':<52} {'record':<10} {'ticks':>6} {'deaths':>7} {'cap':>6} {'score':>8}")
    rows = []
    reference: Optional[float] = None
    for tune in variants:
        record, stats = run_variant(tune, args.jobs, args.timeout)
        value = score(record, stats)
        if reference is None:
            reference = value
        label = tune if tune else "(defaults)"
        rec = f"{record['win']}W{record['loss']}L{record['draw']}D"
        losses = stats.get("_losses") or []
        print(f"{label:<52} {rec:<10} {_mean(stats.get('ticks')):6.0f} "
              f"{_mean(stats.get('deaths')):7.1f} {_mean(stats.get('final_capture')):6.2f} "
              f"{value:8.1f}"
              + (f"   lost/drew vs {' '.join(losses)}" if losses else ""))
        rows.append((value - reference, label, rec))

    print("\nranked by change vs defaults:")
    for delta, label, rec in sorted(rows, key=lambda r: -r[0]):
        marker = "  " if abs(delta) < 0.5 else ("+ " if delta > 0 else "- ")
        print(f"  {marker}{delta:+8.1f}  {label:<52} {rec}")

    os.environ.pop("MM_TUNE", None)


if __name__ == "__main__":
    main()
