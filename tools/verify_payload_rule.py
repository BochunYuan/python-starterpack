"""Settle the payload ambiguity from real match data.

`step_payload` in the shipped engine and the doc comment directly above it describe two
different rules:

  code:    delta = margin.signum() * conf.payload.speed / path_len
           -- one bot pushes exactly as fast as twenty, and any single enemy bot in the ring
              freezes the payload outright.
  comment: a `contest_diff` deadband, then `speed_per_bot` per bot of advantage beyond it,
           capped at `max_speed` -- so each extra body is more speed.

`PayloadConfig` has no field for `contest_diff`, `speed_per_bot` or `max_speed`, which is
already strong evidence, but evidence from the engine we are actually judged by is better.

This measures it. For every tick of a gamelog it counts each team's bots inside
`capture_radius` of the payload centre and records the change in `capture` on the following
tick. Group those deltas by how many bots the pushing side had:

  * every group showing the same |delta| -> the code is the rule
  * |delta| growing with the count      -> the comment is the rule

Run it on any log with real contests in it:

    python3 tools/verify_payload_rule.py /tmp/m4.mmgl
"""

import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load  # noqa: E402


def payload_pos(path: List[Tuple[float, float]], t: float) -> Tuple[float, float]:
    """`config::payload_position` in Python: arc-length walk along the path, mirrored for t<0.

    Reimplemented rather than called over the FFI so this script runs on a log alone, with no
    engine handle and no live match.
    """
    mirrored = t < 0.0
    total = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    remaining = min(abs(t), 1.0) * total
    pos = path[-1]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg = math.dist(a, b)
        if remaining <= seg:
            f = 0.0 if seg == 0.0 else remaining / seg
            pos = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            break
        remaining -= seg
    if mirrored:
        pos = (32.0 - pos[0], 32.0 - pos[1])
    return pos


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: verify_payload_rule.py <log.mmgl> [more...]")
        return

    config, states, _ = load(sys.argv[1])
    for extra in sys.argv[2:]:
        _, more, _ = load(extra)
        states.extend(more)

    path = [(p["x"], p["y"]) for p in config["payload_path"]]
    capture_radius = float(config["payload"]["capture_radius"])
    speed = float(config["payload"]["speed"])
    total_len = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    expected = speed / total_len

    r_sq = capture_radius * capture_radius

    # margin (a_count - b_count) -> observed |delta| values
    by_margin: Dict[int, List[float]] = defaultdict(list)
    # pushing-side count -> observed |delta|, for the uncontested case only
    by_count: Dict[int, List[float]] = defaultdict(list)
    frozen_with_both = 0
    moved_with_both = 0

    for i in range(len(states) - 1):
        cur, nxt = states[i], states[i + 1]
        cap = float(cur.get("capture", 0.0))
        centre = payload_pos(path, cap)

        def count(team: str) -> int:
            n = 0
            for bot in (cur.get(f"fleet_{team}") or {}).values():
                p = bot.get("pos") or {}
                dx = float(p.get("x", 0.0)) - centre[0]
                dy = float(p.get("y", 0.0)) - centre[1]
                if dx * dx + dy * dy <= r_sq:
                    n += 1
            return n

        na, nb = count("a"), count("b")
        delta = float(nxt.get("capture", 0.0)) - cap

        if na and nb:
            if abs(delta) < 1e-9:
                frozen_with_both += 1
            else:
                moved_with_both += 1
            continue

        if not na and not nb:
            continue

        by_margin[na - nb].append(abs(delta))
        by_count[max(na, nb)].append(abs(delta))

    print(f"config: payload.speed={speed}  path_len={total_len}")
    print(f"expected |delta| under the CODE rule (flat): {expected:.9f}")
    print()
    print("--- both teams in the ring (the code says: frozen) ---")
    print(f"  ticks frozen: {frozen_with_both}")
    print(f"  ticks moved : {moved_with_both}")
    if moved_with_both == 0 and frozen_with_both > 0:
        print("  => contested means frozen. Matches the code.")
    elif moved_with_both:
        print("  => the payload moved while contested. Contradicts the code!")
    print()
    print("--- uncontested: |delta| by number of bots pushing ---")
    print("   bots   ticks      mean |delta|    ratio vs expected")
    flat = True
    for n in sorted(by_count):
        vals = by_count[n]
        mean = sum(vals) / len(vals)
        ratio = mean / expected if expected else 0.0
        print(f"  {n:5d}  {len(vals):6d}   {mean:.9f}   {ratio:8.3f}")
        if abs(ratio - 1.0) > 0.02:
            flat = False

    print()
    if flat and len(by_count) > 1:
        print("VERDICT: |delta| is flat across bot counts over "
              f"{len(by_count)} distinct counts.")
        print("  The CODE is the rule: one bot pushes as fast as many.")
        print("  Keep tuning.AMBIGUITY_PAYLOAD_SPEED_SCALES_WITH_BOT_COUNT = False")
    elif len(by_count) <= 1:
        print("INCONCLUSIVE: only one distinct bot count was observed pushing.")
        print("  Run a match where the fleet masses on the payload to get more counts.")
    else:
        print("VERDICT: |delta| varies with bot count.")
        print("  The COMMENT is the rule: mass bots on the payload.")
        print("  Set tuning.AMBIGUITY_PAYLOAD_SPEED_SCALES_WITH_BOT_COUNT = True")


if __name__ == "__main__":
    main()
