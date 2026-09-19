"""Small geometry helpers, in pure Python on purpose.

Every one of these is something the engine could answer over the C ABI, and for a handful of
calls the engine's version is the one to use -- `line_of_sight` and `navigate_to` do work no
Python loop should attempt. These are the other case: the questions cheap enough that the
~1-1.5us of a channel crossing is the expensive part, and which get asked once per bot pair.

A bot pair in a Python loop costs about 0.36us and an engine query 1-1.5us, so anything
asked O(n^2) times belongs here and anything asked once per bot belongs to the engine.
"""

import math

from core._generated.bindings import Vec2


def seg_point_dist_sq(a: Vec2, b: Vec2, p: Vec2) -> float:
    """Squared distance from `p` to the segment `a`->`b`.

    Squared, and taking no square roots, because the callers all compare against a squared
    threshold. The clamp is what makes it a segment rather than an infinite line.
    """
    dx, dy = b.x - a.x, b.y - a.y
    len_sq = dx * dx + dy * dy
    if len_sq <= 0.0:
        return p.dist_sq(a)
    t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq
    if t <= 0.0:
        return p.dist_sq(a)
    if t >= 1.0:
        return p.dist_sq(b)
    nx, ny = a.x + dx * t, a.y + dy * t
    ex, ey = p.x - nx, p.y - ny
    return ex * ex + ey * ey


def disc_blocks(a: Vec2, b: Vec2, centre: Vec2, radius: float) -> bool:
    """Whether a solid disc sits on the segment `a`->`b` somewhere strictly before `b`.

    The blaster's ray stops at the payload and at either deposit, so a shot with a clear
    wall sightline can still be eaten by one of them. `line_of_sight` does not know about
    either -- it checks walls and the arena boundary only -- so this is the other half of
    "can this shot actually reach".

    A disc that contains `b` itself does not count as blocking: that is the target standing
    against the payload, and the blast lands on the hull either way.
    """
    if b.dist_sq(centre) <= radius * radius:
        return False
    return seg_point_dist_sq(a, b, centre) <= radius * radius


def ring_points(centre: Vec2, radius: float, count: int, phase_deg: float = 0.0):
    """`count` points evenly spaced on a circle. Used to generate candidate standing spots."""
    out = []
    for i in range(count):
        a = math.radians(phase_deg + 360.0 * i / count)
        out.append(Vec2(centre.x + math.cos(a) * radius, centre.y + math.sin(a) * radius))
    return out


def spread_apart(spots, min_gap: float):
    """Thin a list of points so no two are within `min_gap`, keeping earlier ones.

    Standing spots are chosen for a whole fleet at once, and two bots on spots closer than
    twice the blast radius are one target for one shot. Greedy and O(k^2) over a short list,
    which is fine off the clock.
    """
    gap_sq = min_gap * min_gap
    kept = []
    for spot in spots:
        if all(spot.dist_sq(other) > gap_sq for other in kept):
            kept.append(spot)
    return kept
