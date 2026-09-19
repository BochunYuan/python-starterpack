"""Generate a panel of reference opponents to measure a strategy change against.

Why a panel and not repeated matches: the engine is effectively deterministic for a given
pair of bots and seats. `eval_tick` shuffles and `closer` coin-flips, but every step that
consumes the order is written to be order-independent (shots resolve against the pre-damage
state, heals accumulate two-phase), so the shuffle is unobservable and the same pairing
replays identically. Running one matchup twenty times produces one data point twenty times --
measured directly: a 12-game and a 20-game run of the same pairing returned byte-identical
averages.

Signal has to come from *variety* instead. Each opponent here plays a different way, so a
change that helps against one and loses to another shows up as a mixed record rather than
hiding inside an aggregate. That also mirrors the tournament, which is a round robin.

    python3 tools/panel_bots.py            # writes /tmp/mm-panel/<name>/run

Each generated bot shares the main bot's `core/` by symlink and points `MM_NATIVE_LIB` at the
already-built native library, so a panel costs kilobytes rather than a rebuild each.
"""

import os
import shutil
import stat
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
NATIVE = os.path.join(BOT, "native", "target", "release")


PREAMBLE = '''"""Reference opponent -- a test fixture, not a strategy anyone should submit."""

from . import *
'''


# Every opponent shares the same shape: pick a destination and a target per bot, fire when the
# shot can land. What differs is the destination policy, which is the whole point of the panel.
COMMON = '''

def _nearest(frm, fleet):
    best = None
    for bot in fleet:
        if best is None or frm.dist_sq(bot.pos) < frm.dist_sq(best.pos):
            best = bot
    return best


def _shoot(conf, state, slot, bot, enemy):
    """Aim at `enemy` and fire when the shot is in range with a sightline."""
    if enemy is None:
        return False
    slot.turn_action = turn_towards(enemy.pos)
    live = (bot.pos.dist(enemy.pos) <= conf.bot.blaster_range
            and bot.next_fire_tick <= state.tick
            and line_of_sight(bot.pos, enemy.pos))
    slot.special_action = SpecialAction.Battle(fire=live)
    return live


def _mine(conf, slot, bot, deposit):
    slot.turn_action = turn_towards(deposit)
    slot.special_action = SpecialAction.Extractor(mine=True)


def _spend(conf, state, action, want):
    """Standard fabricator handling: build `want`, and rush whenever it is affordable."""
    action.fabricator_next = int(want)
    endgame = state.tick >= conf.max_ticks - conf.endgame_ticks
    action.rush_order = (not endgame
                         and not state.fleet_me.is_full()
                         and state.fabricator_me.tokens >= conf.fabricator.rush_cost)
'''


RUSHER = '''

def get_strategy(team: int) -> Strategy:
    return rush


def rush(state: GameState) -> FleetAction:
    """All-in aggression: every body walks at the nearest enemy and shoots it.

    The opponent that punishes a passive fleet. It concedes the economy entirely and wins by
    arriving first with more guns.
    """
    conf = get_config()
    action = FleetAction.new()
    payload = state.payload_pos()

    for bot in state.fleet_me:
        slot = action.bots[bot.id]
        enemy = _nearest(bot.pos, state.fleet_other)
        _shoot(conf, state, slot, bot, enemy)
        dest = enemy.pos if enemy is not None else payload
        slot.move_action = move_bot(navigate_to(bot.pos, dest))

    _spend(conf, state, action, BotClass.Battle)
    return action
'''


MASSER = '''

def get_strategy(team: int) -> Strategy:
    return mass


def mass(state: GameState) -> FleetAction:
    """Everything stacks in the capture ring.

    This is the opponent implied by the *other* reading of `step_payload` -- the doc comment's
    per-bot speed scaling. Even though that rule is not the one the engine implements, the
    behaviour is a real threat: a body in the ring freezes our push whatever the speed rule
    is, so this bot is hard to make progress against and is the natural test of whether our
    anchor-and-hunt plan can actually clear a contested ring.
    """
    conf = get_config()
    action = FleetAction.new()
    payload = state.payload_pos()

    for bot in state.fleet_me:
        slot = action.bots[bot.id]
        enemy = _nearest(bot.pos, state.fleet_other)
        _shoot(conf, state, slot, bot, enemy)
        # Stop once inside the ring; the payload itself is solid so collision settles them.
        if bot.pos.dist(payload) > conf.payload.capture_radius * 0.6:
            slot.move_action = move_bot(navigate_to(bot.pos, payload))

    _spend(conf, state, action, BotClass.Battle)
    return action
'''


TURTLE = '''

def get_strategy(team: int) -> Strategy:
    return turtle


def turtle(state: GameState) -> FleetAction:
    """Defend the home goal and mine; never contest the centre.

    Deliberately exploits the freeze rule from the defending side: the payload path ends at
    our own goal, so a body parked there stops the match ever being lost outright, and the
    clock then decides it on the tiebreak chain. The opponent that punishes a bot which cannot
    finish a push.
    """
    conf = get_config()
    action = FleetAction.new()
    home = payload_pos(-1.0)
    deposit = state.deposit_me.pos

    miners = 0
    for bot in state.fleet_me:
        slot = action.bots[bot.id]
        if bot.class_ == BotClass.Extractor:
            miners += 1
            spot = deposit + Vec2(0.0, conf.deposit.radius + conf.bot.radius * 3.0)
            if bot.pos.dist(spot) > conf.bot.radius:
                slot.move_action = move_bot(navigate_to(bot.pos, spot))
            _mine(conf, slot, bot, deposit)
            continue

        enemy = _nearest(bot.pos, state.fleet_other)
        _shoot(conf, state, slot, bot, enemy)
        # Hold a loose shell around the goal, spread by bot id so they do not stack.
        offset = Vec2.from_angle_deg(bot.id * 47.0) * (conf.payload.capture_radius * 0.8)
        post = home + offset
        if bot.pos.dist(post) > conf.bot.radius * 2.0:
            slot.move_action = move_bot(navigate_to(bot.pos, post))

    want = BotClass.Extractor if miners < 6 else BotClass.Battle
    _spend(conf, state, action, want)
    return action
'''


ECONOMIST = '''

def get_strategy(team: int) -> Strategy:
    return economy


def economy(state: GameState) -> FleetAction:
    """Saturate both deposits, guard the miners, ignore the payload until late.

    The opponent that wins the long game if we under-build the economy: its income converts
    into a bigger fleet, and a bigger fleet takes the ring whenever it decides to.
    """
    conf = get_config()
    action = FleetAction.new()
    payload = state.payload_pos()
    endgame = state.tick >= conf.max_ticks - conf.endgame_ticks

    miners = 0
    for bot in state.fleet_me:
        slot = action.bots[bot.id]
        if bot.class_ == BotClass.Extractor:
            # Alternate deposits by id so the fleet does not pile onto one shared pool.
            deposit = (state.deposit_me if bot.id % 3 else state.deposit_other).pos
            miners += 1
            ring = conf.deposit.radius + conf.bot.radius * 3.0 + (bot.id % 4) * 0.4
            spot = deposit + Vec2.from_angle_deg(bot.id * 53.0) * ring
            if bot.extracting is None and bot.pos.dist(spot) > conf.bot.radius:
                slot.move_action = move_bot(navigate_to(bot.pos, spot))
            _mine(conf, slot, bot, deposit)
            continue

        enemy = _nearest(bot.pos, state.fleet_other)
        _shoot(conf, state, slot, bot, enemy)
        # Escort the economy early, take the payload once the fabricator is done.
        if endgame:
            dest = payload
        elif enemy is not None and bot.pos.dist(enemy.pos) <= conf.bot.blaster_range * 1.5:
            dest = enemy.pos
        else:
            dest = state.deposit_me.pos
        slot.move_action = move_bot(navigate_to(bot.pos, dest))

    want = BotClass.Extractor if miners < 14 else BotClass.Battle
    _spend(conf, state, action, want)
    return action
'''


PANEL = {
    "rusher": RUSHER,
    "masser": MASSER,
    "turtle": TURTLE,
    "economist": ECONOMIST,
}


def build_bot(root: str, name: str, body: str, strategy_src: str = "") -> str:
    """Write one opponent and return the path to its run script."""
    path = os.path.join(root, name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(os.path.join(path, "strategy"), exist_ok=True)

    # `core/` is shared by symlink -- it is generated code, identical for every bot, and
    # `MM_NATIVE_LIB` in the run script is what makes the native library resolvable from here.
    os.symlink(os.path.join(BOT, "core"), os.path.join(path, "core"))
    shutil.copy(os.path.join(BOT, "__main__.py"), os.path.join(path, "__main__.py"))
    shutil.copy(
        os.path.join(BOT, "strategy", "__init__.py"),
        os.path.join(path, "strategy", "__init__.py"),
    )

    with open(os.path.join(path, "strategy", "main.py"), "w") as handle:
        handle.write(strategy_src or (PREAMBLE + COMMON + body))

    lib = ""
    for candidate in ("libmm_python_native.dylib", "libmm_python_native.so",
                      "mm_python_native.dll"):
        full = os.path.join(NATIVE, candidate)
        if os.path.exists(full):
            lib = full
            break

    run = os.path.join(path, "run")
    with open(run, "w") as handle:
        handle.write("#!/bin/sh\n")
        if lib:
            handle.write(f'MM_NATIVE_LIB="{lib}"\nexport MM_NATIVE_LIB\n')
        handle.write(f'exec python3 "{path}/__main__.py" "$1"\n')
    os.chmod(run, os.stat(run).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return run


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mm-panel"
    os.makedirs(root, exist_ok=True)

    for name, body in PANEL.items():
        run = build_bot(root, name, body)
        print(f"built {name}: {run}")

    # The shipped starter strategy, with its team-1 `do_nothing` replaced so it plays both
    # seats. Without that swap it forfeits one seat and stops being a reference at all.
    starter = None
    cache = os.path.join(BOT, ".mm", "cache")
    if os.path.isdir(cache):
        for entry in sorted(os.listdir(cache)):
            candidate = os.path.join(cache, entry, "strategy", "main.py")
            if "starterpack" in entry and os.path.exists(candidate):
                starter = candidate
                break
    if starter:
        src = open(starter).read()
        src = src.replace(
            '''    if team == 0:
        print("Hello! I am team A (on the bottom left)")
        return basic_strategy
    else:
        print("Hello! I am team B (on the top right)")
        return do_nothing''',
            "    return basic_strategy",
        )
        run = build_bot(root, "starter", "", strategy_src=src)
        print(f"built starter: {run}")
    else:
        print("warning: starter strategy not found in .mm/cache; skipping", file=sys.stderr)


if __name__ == "__main__":
    main()
