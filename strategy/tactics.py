"""Per-class per-tick orders, and the fleet-wide fire plan that has to come before them.

Two things live here that are easy to get wrong if each bot decides alone:

**Fire allocation.** A bot that takes a blast becomes invulnerable for
`base_invulnerability_ticks`, and `step_blasters` skips an invulnerable bot entirely. Both
numbers are absolute ticks set at the moment of the hit, so two of our bots firing at the
same enemy on the same tick land one blast between them -- the second is skipped and its
`blaster_cooldown` is spent for nothing. So fire is allocated across the fleet, at most one
shooter per enemy per tick, and a shooter with no unclaimed target holds its shot rather
than wasting the cooldown.

**Spacing.** The blast catches every enemy whose centre is within
`base_blaster_splash_radius + radius` of the impact point, so two of our own bots closer
together than twice that are one target for one shot. Every move order gets a separation
nudge for that reason.

Everything here reads `World` and writes into a `FleetAction`; nothing re-derives state.
"""

import math
from typing import Dict, List, Optional, Tuple

from core.channel import line_of_sight, move_bot, navigate_to, turn_towards
from core._generated.bindings import (
    BotClass,
    BotState,
    FleetAction,
    SpecialAction,
    Vec2,
)

from . import tuning
from .geom import disc_blocks
from .world import World


# ---------------------------------------------------------------------------------------
# movement
# ---------------------------------------------------------------------------------------


def blend_move(nav: Vec2, nudge: Vec2, weight: float) -> Vec2:
    """Combine a `navigate_to` delta with a separation nudge, preserving arrival behaviour.

    `navigate_to` returns `(target - pos) / speed`, so its length is "ticks of travel left".
    `MoveAction::sanitize` only normalizes a direction *longer* than one -- a shorter one is
    a request to move proportionally slower, which is what lets a bot stop exactly on its
    target instead of oscillating across it.

    So: while there is more than a tick of travel left, go at full speed in the blended
    direction. Once inside the last tick, keep the short vector and let the bot settle,
    since a nudge applied there would push it off the spot it just reached.
    """
    n = nav.norm()
    if n <= 1.0:
        return nav
    unit = Vec2(nav.x / n, nav.y / n)
    if nudge.x == 0.0 and nudge.y == 0.0:
        return unit
    blended = Vec2(unit.x + nudge.x * weight, unit.y + nudge.y * weight)
    m = blended.norm()
    return unit if m <= 0.0 else Vec2(blended.x / m, blended.y / m)


def separation_nudge(world: World, bot: BotState) -> Vec2:
    """A unit push away from allies that are close enough to share a blast.

    Only allies inside the danger radius contribute, so in open play this is the zero vector
    and costs nothing. Enemies are deliberately not considered: being near an enemy is how
    a fight happens, and their blast radius is centred on wherever their ray stops anyway.
    """
    limit = world.static.blast_radius * tuning.SPACING_SAFETY_MULTIPLE
    limit_sq = limit * limit
    ax = ay = 0.0
    for other in world.mine:
        if other.id == bot.id:
            continue
        d_sq = bot.pos.dist_sq(other.pos)
        if d_sq >= limit_sq or d_sq <= 0.0:
            continue
        # Weight by how deep the overlap is, so the nudge fades out at the limit rather
        # than switching on and off and making the bot jitter at the boundary.
        d = math.sqrt(d_sq)
        strength = (limit - d) / limit
        ax += (bot.pos.x - other.pos.x) / d * strength
        ay += (bot.pos.y - other.pos.y) / d * strength
    if ax == 0.0 and ay == 0.0:
        return Vec2(0.0, 0.0)
    n = math.hypot(ax, ay)
    return Vec2(ax / n, ay / n)


def go_to(world: World, bot: BotState, dest: Vec2, spaced: bool = True) -> Vec2:
    """The move vector to walk toward `dest` around walls, kept clear of our own bots."""
    nav = navigate_to(bot.pos, dest)
    nudge = separation_nudge(world, bot) if spaced else Vec2(0.0, 0.0)
    return blend_move(nav, nudge, tuning.SPACING_WEIGHT)


# ---------------------------------------------------------------------------------------
# fire allocation
# ---------------------------------------------------------------------------------------


def _predict(bot: BotState) -> Vec2:
    """Where a bot will be after this tick's movement.

    `bot.vel` is the velocity the engine recorded for last tick, which is the best estimate
    available of what it is about to do again. Movement resolves before `step_blasters`, so
    the shot is checked against post-move positions -- predicting is not optional if the fire
    gate is to mean anything.
    """
    return Vec2(bot.pos.x + bot.vel.x, bot.pos.y + bot.vel.y)


def _shot_lands(world: World, origin: Vec2, target: Vec2, dist: float) -> bool:
    """Whether a ray from `origin` toward `target` would put its blast on `target`.

    Three gates, cheapest first, because the last one is an engine call:

    1. Inside `blaster_range` -- the ray is capped there and bursts in the air past it.
    2. Not eaten by the payload or either deposit, which stop the ray exactly like a wall.
       `line_of_sight` knows nothing about those -- it tests wall tiles and the arena boundary
       only -- so they need their own check, and they are the reason a nominally clear shot can
       still land on nothing.
    3. A clear wall sightline.

    There is deliberately no separate aiming check. One was written -- comparing the ray's
    perpendicular miss against the target's hull -- and it was a no-op: `plan_fire` derives the
    angle *from* the target, so the miss it measured was always zero to sixteen digits, against a
    threshold of 0.21. A gate that cannot reject is worse than no gate, because it reads as
    assurance. What actually bounds the aim is the swing test in `plan_fire`, which drops any
    target more than one tick of rotation off the barrel. Measured end to end: 91% of shots
    taken land on something.
    """
    conf = world.conf
    if dist > conf.bot.blaster_range:
        return False
    if disc_blocks(origin, target, world.payload, conf.payload.radius):
        return False
    for key in (0, 1):
        if disc_blocks(origin, target, world.deposit_pos(key), conf.deposit.radius):
            return False
    return line_of_sight(origin, target)


def _threat_score(world: World, enemy: BotState) -> float:
    """How much we want this enemy dead, before distance is considered.

    An enemy standing in the capture ring is the only thing that can stop our push, and
    killing it is worth more than any amount of damage elsewhere. After that, finishing a
    bot that one more blast kills beats starting on a fresh one -- a kill removes its output
    permanently, where damage on a full-health bot is often just healed back.
    """
    score = 0.0
    if enemy.pos.dist_sq(world.payload) <= world.static.capture_outer ** 2:
        score += 100.0
    if enemy.health <= world.conf.bot.blaster_damage:
        score += 60.0
    # Prefer the hurt over the healthy, scaled so it never outweighs a kill or an anchor.
    score += (world.conf.bot.health - enemy.health) * 2.0
    if enemy.class_ == BotClass.Healer:
        score += 25.0
    elif enemy.class_ == BotClass.Extractor:
        score += 10.0
    return score


def plan_fire(world: World, shooters: List[BotState]) -> Dict[int, BotState]:
    """Assign at most one shooter per enemy, and only where the shot actually lands.

    Returns bot id -> the enemy to aim at. A shooter absent from the result has no shot worth
    taking and should hold: a wasted trigger pull is `blaster_cooldown` ticks of silence.

    One enemy per shooter because damage does not stack within a tick. A bot that takes a blast
    is invulnerable for `base_invulnerability_ticks`, set at the moment of the hit, and
    `step_blasters` skips an invulnerable bot entirely -- so two of our bots firing at the same
    enemy land one blast between them and the second cooldown is spent for nothing.

    Deliberately *not* splash-aware. The blast catches every enemy within
    `base_blaster_splash_radius + radius` of the impact point, so aiming to clip several at once
    looks like free damage, and a version of this function that clustered the enemies and scored
    each shot by how many it would catch was written and measured. It did not pay: enemies were
    damaged at 1.05 bots per shot versus 1.01, kill counts were *identical* against all five
    panel opponents, and the clustering pass raised the mean per-tick charge from 167 to 187 and
    the worst from 588 to 934 against a refill of 600. The splash radius is simply small next to
    the distances bots actually sit at, so the blast already collects a second body whenever one
    happens to be there, and planning for it adds cost without adding kills.

    The pair scan is plain arithmetic at roughly 0.36us a pair. The engine call
    (`line_of_sight`) happens at most once per accepted shot and never inside the pair loop --
    an engine query per pair is the one shape that reliably blows the budget.
    """
    if not shooters or not world.theirs:
        return {}

    conf = world.conf
    reach_sq = conf.bot.blaster_range ** 2
    turn_limit = conf.bot.turn_speed

    enemies = [e for e in world.theirs if world.vulnerable(e)]
    if not enemies:
        return {}

    # Post-move positions. Movement resolves before `step_blasters`, so a gate checked against
    # current positions is a tick stale and rejects shots that would have landed.
    predicted = [(e, _predict(e), _threat_score(world, e)) for e in enemies]

    candidates: List[Tuple[float, BotState, BotState, Vec2, float, float]] = []
    for shooter in shooters:
        here = _predict(shooter)
        for enemy, there, threat in predicted:
            d_sq = here.dist_sq(there)
            if d_sq > reach_sq:
                continue
            # Can the barrel get there this tick? `TurnAction::TargetPosition` turns at most
            # `turn_speed` degrees and the shot resolves after the turn, so a target beyond one
            # tick of rotation cannot be hit yet -- though it is still worth turning toward.
            want = (there - here).angle_deg()
            if abs(_signed_degrees(want - shooter.angle)) > turn_limit:
                continue
            d = math.sqrt(d_sq)
            candidates.append((threat - d, shooter, enemy, there, d, want))

    if not candidates:
        return {}

    candidates.sort(key=lambda c: -c[0])

    plan: Dict[int, BotState] = {}
    claimed: set = set()
    for _, shooter, enemy, there, d, want in candidates:
        if shooter.id in plan or enemy.id in claimed:
            continue
        here = _predict(shooter)
        if not _shot_lands(world, here, there, d):
            continue
        plan[shooter.id] = enemy
        claimed.add(enemy.id)
    return plan


def _signed_degrees(deg: float) -> float:
    """`deg` wrapped into [-180, 180). Mirrors the engine's own `diff_degrees` result."""
    d = (deg + 180.0) % 360.0 - 180.0
    return d


# ---------------------------------------------------------------------------------------
# per-class orders
# ---------------------------------------------------------------------------------------


def order_battle(
    world: World,
    action: FleetAction,
    bot: BotState,
    target: Optional[BotState],
    station: Optional[Vec2],
) -> None:
    """Orders for one battle bot: where to stand, what to face, whether to pull the trigger.

    `target` comes from `plan_fire` and is `None` when no shot of ours should land on it this
    tick. `station` is a place to be -- the capture ring for an anchor, `None` for a hunter,
    which closes on its target instead.
    """
    slot = action.bots[bot.id]
    conf = world.conf

    aim: Optional[Vec2] = None
    if target is not None:
        aim = _predict(target)
    else:
        # Nothing shootable: still face the nearest enemy, so the barrel is already pointed
        # when one becomes shootable. Turning is free and costs no cooldown.
        near = world.nearest(bot.pos, world.theirs)
        if near is not None:
            aim = near.pos

    if station is not None:
        slot.move_action = move_bot(go_to(world, bot, station))
    elif aim is not None:
        # Hold the engagement band only while we actually have a shot from here. Standing at
        # the edge of our range is also the edge of theirs, so the band is worth holding --
        # but the band is a *distance*, and on this map a target 8 units away is usually
        # behind a wall. A bot that holds the band without a sightline stands still and never
        # fires, which is how a fleet loses without ever fighting.
        #
        # So: keep the band when the shot is live, and close when it is not. Closing is what
        # clears the wall, and the payload -- which stops a ray exactly like a wall -- is the
        # other reason a nominally in-range target cannot be hit from standoff.
        band = conf.bot.blaster_range * tuning.ENGAGE_RANGE_FRACTION
        d = bot.pos.dist(aim)
        hold = d <= band and target is not None
        if not hold:
            slot.move_action = move_bot(go_to(world, bot, aim))
        else:
            nudge = separation_nudge(world, bot)
            if nudge.x != 0.0 or nudge.y != 0.0:
                slot.move_action = move_bot(nudge)

    if aim is not None:
        slot.turn_action = turn_towards(aim)

    fire = target is not None and world.can_fire(bot)
    slot.special_action = SpecialAction.Battle(fire=fire)


def order_healer(
    world: World, action: FleetAction, bot: BotState, target: Optional[BotState]
) -> None:
    """Orders for one healer: stay in reach of its charge, face it, channel.

    The heal only lands when the target is inside `base_heal_range` (centre to centre) *and*
    within half of `base_heal_arc_deg` of the healer's facing, with walls blocking. So the
    facing matters as much as the position, and a healer that is turning is a healer that is
    not yet healing -- which is why it parks well inside the range rather than at its edge.
    """
    slot = action.bots[bot.id]
    conf = world.conf

    if target is None:
        # No one to heal: shadow the fight from behind so the next casualty is in reach.
        anchor = world.nearest(bot.pos, world.battle) or world.nearest(bot.pos, world.mine)
        if anchor is not None and anchor.id != bot.id:
            keep = conf.bot.base_heal_range * 0.5
            if bot.pos.dist(anchor.pos) > keep:
                slot.move_action = move_bot(go_to(world, bot, anchor.pos))
            slot.turn_action = turn_towards(anchor.pos)
        slot.special_action = SpecialAction.Healer(fire=False, target=bot.id)
        return

    # Sit at half of the reach: close enough that the target drifting does not break the
    # channel, far enough that we are not inside its blast radius.
    keep = conf.bot.base_heal_range * 0.5
    d = bot.pos.dist(target.pos)
    if d > keep:
        slot.move_action = move_bot(go_to(world, bot, target.pos))
    else:
        nudge = separation_nudge(world, bot)
        if nudge.x != 0.0 or nudge.y != 0.0:
            slot.move_action = move_bot(nudge)

    slot.turn_action = turn_towards(target.pos)
    slot.special_action = SpecialAction.Healer(fire=True, target=target.id)


def order_extractor(
    world: World, action: FleetAction, bot: BotState, spot: Vec2, deposit: Vec2
) -> None:
    """Orders for one extractor: get to a spot that sees the deposit, face it, mine.

    Slots are sticky -- `step_extractors` re-seats incumbents before admitting anyone new --
    so the one thing an extractor must not do is stop qualifying. Once it is mining it holds
    still and keeps facing the deposit; drifting or turning away opens its slot to the enemy
    on that same tick, and it comes back as a newcomer at the end of the queue.
    """
    slot = action.bots[bot.id]

    mining = bot.extracting is not None
    if not mining:
        # Deliberately unspaced: the mining spots were chosen apart from each other at
        # startup, and a separation nudge here would fight the assignment and could walk a
        # bot out of its sightline.
        nav = navigate_to(bot.pos, spot)
        slot.move_action = move_bot(nav)

    slot.turn_action = turn_towards(deposit)
    slot.special_action = SpecialAction.Extractor(mine=True)
