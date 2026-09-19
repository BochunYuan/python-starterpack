"""The strategy the engine calls once per tick.

Layers, cheapest first, each reading the one below and never re-deriving it:

    world.Static    built once after the handshake, before tick 0 -- not charged
    world.World     one pass over both fleets per tick, no engine queries
    roles.Brain     roles and the build queue, refreshed on a cadence and carried over
    tactics         per-class orders, plus the fleet-wide fire plan

The shape of the plan follows the shipped `step_payload`: push speed is flat, so one bot moves
the payload as fast as twenty, and any single enemy bot in the ring freezes it. That makes the
capture ring the only place the match can be decided -- the enemy has to come there too -- so
the fleet masses there (`PAYLOAD_ANCHOR_COUNT`, measured head-to-head) rather than holding a
thin anchor and hunting elsewhere. Within the ring bots still stand `blast_radius` apart,
because one blast catches everything inside that radius.

See `tuning.AMBIGUITY_PAYLOAD_SPEED_SCALES_WITH_BOT_COUNT` for the rule this rests on and
`tools/verify_payload_rule.py` for the measurement that settled it.
"""

import sys
from typing import Optional

from . import tuning
from .roles import ANCHOR, HUNTER, MINER, SUPPORT, Brain
from .tactics import (
    go_to,
    order_battle,
    order_extractor,
    order_healer,
    plan_fire,
)
from .world import Static, World

from core.channel import Strategy, get_budget, move_bot, navigate_to, turn_towards
from core._generated.bindings import (
    BotClass,
    FleetAction,
    GameState,
    SpecialAction,
    Vec2,
)


def get_strategy(team: int) -> Strategy:
    """Build the strategy for this match.

    Runs after the handshake and before the first tick, so everything set up here is off the
    clock -- which is the whole reason `Static` and the mining spots are resolved now rather
    than lazily on tick 0.

    Both sides get the same strategy. The engine mirrors the world for team B (positions,
    angles, velocities, and the sign of `capture`), so there is nothing for a side to
    specialise in and a side-specific bot would only be two things to debug.
    """
    brain: Optional[Brain] = None
    static: Optional[Static] = None
    faults = [0]

    def strategy(state: GameState) -> FleetAction:
        nonlocal brain, static

        # An uncaught exception here does not cost one tick -- it costs the match. Measured:
        # the process dies, the engine reports "response timed out", and the fleet then sits
        # inert for every remaining tick while the opponent walks the payload in. That is a
        # guaranteed loss from a single bad frame, so the whole tick is wrapped and any failure
        # degrades to orders that are merely poor instead.
        try:
            if static is None:
                # `get_config()` needs the handshake, which has happened by the time the first
                # tick arrives. Tick 0 is not charged, so the map scan here is free.
                static = Static()
                brain = Brain(static)
                world = World(static, state)
                brain.init_spots(world)
            else:
                world = World(static, state)

            if brain is None:
                return _fallback(state)
            return play(brain, world)
        except Exception as exc:  # noqa: BLE001 -- a crash must never reach the channel
            faults[0] += 1
            # Printed, because stdout and stderr are forwarded into the gamelog and a silent
            # fallback is indistinguishable from a bot that is merely playing badly. Rate
            # limited: a fault that repeats every tick would otherwise write 9000 lines and
            # the printing itself would start costing budget.
            if faults[0] <= 3 or faults[0] % 500 == 0:
                print(f"[fault #{faults[0]}] tick {state.tick}: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            try:
                return _fallback(state)
            except Exception:  # noqa: BLE001 -- last resort, must return something valid
                return FleetAction.new()

    return strategy


def _fallback(state: GameState) -> FleetAction:
    """Orders that need nothing but the state: walk to the payload, shoot what is in front.

    Deliberately not "do nothing". An empty action concedes the payload, and conceding the
    payload loses outright, whereas a fleet sitting in the capture ring at least freezes an
    enemy push and holds the first tiebreak. Uses only `navigate_to`, `turn_towards` and plain
    arithmetic, so it stays valid even if every cached structure above it is broken.
    """
    action = FleetAction.new()
    payload = state.payload_pos()

    for bot in state.fleet_me:
        slot = action.bots[bot.id]
        slot.move_action = move_bot(navigate_to(bot.pos, payload))

        nearest = None
        best = 0.0
        for enemy in state.fleet_other:
            d = bot.pos.dist_sq(enemy.pos)
            if nearest is None or d < best:
                nearest, best = enemy, d

        if nearest is not None:
            slot.turn_action = turn_towards(nearest.pos)
        else:
            slot.turn_action = turn_towards(payload)

        if bot.class_ == BotClass.Extractor:
            slot.special_action = SpecialAction.Extractor(mine=True)
        elif bot.class_ == BotClass.Healer:
            slot.special_action = SpecialAction.Healer(fire=False, target=bot.id)
        else:
            slot.special_action = SpecialAction.Battle(
                fire=nearest is not None and bot.next_fire_tick <= state.tick
            )

    return action


def play(brain: Brain, world: World) -> FleetAction:
    """One tick of orders for the whole fleet."""
    action = FleetAction.new()

    # An overspend is a debt repaid in ticks where the entire fleet does nothing, which costs
    # far more than one cheap tick does. Below the floor: keep last tick's roles, skip the
    # optional work, still give every bot something to do.
    cheap = get_budget().remaining < tuning.BUDGET_FLOOR

    roles = brain.assign(world, cheap)

    # --- where the anchors stand -------------------------------------------------------
    anchor_ids = [bid for bid, role in roles.items() if role == ANCHOR]
    stations = brain.anchor_stations(world, len(anchor_ids))
    station_of = {bid: stations[i] for i, bid in enumerate(sorted(anchor_ids))
                  if i < len(stations)}

    # --- who shoots what ---------------------------------------------------------------
    shooters = [bot for bot in world.battle if world.can_fire(bot)]
    fire_plan = plan_fire(world, shooters) if shooters else {}

    # --- who heals whom ---------------------------------------------------------------
    heal_plan = {} if cheap else brain.heal_targets(world)

    # --- orders ------------------------------------------------------------------------
    for bot in world.mine:
        role = roles.get(bot.id, HUNTER)

        if role == MINER:
            spot = brain.spot_of.get(bot.id)
            if spot is None:
                spot = bot.pos
            # Face whichever deposit the spot was chosen for: the mining ray follows the
            # bot's facing, and a miner pointed at the wrong disc earns nothing.
            deposit = _nearest_deposit(world, spot)
            order_extractor(world, action, bot, spot, deposit)
            continue

        if role == SUPPORT:
            order_healer(world, action, bot, heal_plan.get(bot.id))
            continue

        if bot.class_ == BotClass.Extractor:
            # A surplus extractor sent to the ring. It cannot shoot, but a body in the ring
            # freezes an enemy push, and that is worth more than mining a full deposit.
            _order_body(world, action, bot, station_of.get(bot.id) or world.payload)
            continue

        if bot.class_ == BotClass.Healer:
            order_healer(world, action, bot, heal_plan.get(bot.id))
            continue

        order_battle(world, action, bot, fire_plan.get(bot.id), station_of.get(bot.id))

    # --- fabricator -------------------------------------------------------------------
    next_class = brain.next_class(world)
    action.fabricator_next = int(next_class)
    action.rush_order = brain.want_rush(world)

    # The natural build lands on `next_bot_creation` and a rush lands now, so either way a
    # bot of this class is what arrives next. Advancing the opening queue here keeps it in
    # step with bodies rather than with ticks.
    if action.rush_order or world.state.fabricator_me.next_bot_creation <= world.tick:
        brain.consume_build(world, next_class)

    return action


def _order_body(world: World, action: FleetAction, bot, dest: Vec2) -> None:
    """Move a non-combatant to a place and have it face the fight. No special action."""
    slot = action.bots[bot.id]
    slot.move_action = move_bot(go_to(world, bot, dest))
    near = world.nearest(bot.pos, world.theirs)
    if near is not None:
        slot.turn_action = turn_towards(near.pos)
    # An extractor told to mine while standing in the ring costs nothing and occasionally
    # pays: the ray only needs a sightline, and bots do not block it.
    slot.special_action = SpecialAction.Extractor(mine=True)


def _nearest_deposit(world: World, to: Vec2) -> Vec2:
    a = world.deposit_pos(0)
    b = world.deposit_pos(1)
    return a if to.dist_sq(a) <= to.dist_sq(b) else b
