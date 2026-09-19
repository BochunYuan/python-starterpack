"""One cheap pass over the tick's state, turned into the facts every other module reads.

Built fresh each tick and thrown away. Nothing here decides anything -- it exists so that
the tactics modules never walk `fleet_me` a second time to answer a question this already
answered, and never call an engine query inside a loop over pairs.

Cost: two passes over at most 32 bots each, no engine queries. That is on the order of a
hundredth of a tick's refill, which is what buys the tactics layer room to spend.

`Static` is the other half: the parts of the world that cannot change during a match. It is
built once in `get_strategy`, which runs after the handshake and before the first tick --
outside the per-tick charge entirely -- so anything that only depends on the config belongs
there rather than here.
"""

from typing import Dict, List, Optional

from core.channel import get_config, path_length, payload_pos
from core._generated.bindings import (
    MAP_SIZE,
    BotClass,
    BotState,
    GameConfig,
    GameState,
    Team,
    Vec2,
)


class Static:
    """Everything derivable from the config alone, computed once off the clock."""

    def __init__(self) -> None:
        conf = get_config()
        self.conf: GameConfig = conf

        # `capture` runs -1 (our goal) to +1 (theirs), so the two ends of the track are
        # just the extremes of the same function `state.payload_pos()` uses.
        self.goal_mine: Vec2 = payload_pos(-1.0)
        self.goal_theirs: Vec2 = payload_pos(1.0)
        self.path_centre: Vec2 = payload_pos(0.0)

        # The tick the fabricator dies. After this nothing is built and tokens are dead
        # weight, so it is the deadline every spending decision is measured against.
        self.endgame_tick: int = conf.max_ticks - conf.endgame_ticks

        # One blast catches every enemy whose *centre* is within this of the impact point:
        # `step_blasters` compares `bot.pos.dist_sq(point)` against
        # `base_blaster_splash_radius + radius`. The same number is what makes two of our
        # own bots standing closer than twice this a single target.
        self.blast_radius: float = (
            conf.bot.base_blaster_splash_radius + conf.bot.radius
        )

        # A bot pushes the payload when its centre is within `capture_radius` of the payload
        # centre, but it cannot stand closer than this -- the payload is solid, and
        # `handle_collision` shoves bots out to `payload.radius + bot.radius`. So the ring a
        # contesting bot actually occupies is an annulus, not a disc.
        self.capture_inner: float = conf.payload.radius + conf.bot.radius
        self.capture_outer: float = conf.payload.capture_radius

        # Ticks for the payload to travel the whole track, end to end, uncontested. The
        # length of the match measured in the only currency that wins it outright.
        path_len = 0.0
        pts = list(conf.payload_path)
        for a, b in zip(pts, pts[1:]):
            path_len += a.dist(b)
        self.path_half_length: float = path_len
        self.capture_per_tick: float = (
            conf.payload.speed / path_len if path_len > 0.0 else 0.0
        )

        # Sustained damage of one battle bot, and what one healer cancels. The engine's own
        # config comment ties these together on purpose: `blaster_damage / blaster_cooldown`
        # is exactly `heal_per_tick` in the shipped rulebook, so a lone battle bot cannot
        # out-damage a lone healer and fights are decided by how many of each are in range.
        self.dps_per_battle: float = (
            conf.bot.blaster_damage / conf.bot.blaster_cooldown
            if conf.bot.blaster_cooldown > 0
            else conf.bot.blaster_damage
        )
        self.hps_per_healer: float = conf.bot.heal_per_tick

        # Shots to kill from full, which is what target selection is really counting.
        self.shots_to_kill: int = (
            1
            if conf.bot.blaster_damage <= 0.0
            else int(-(-conf.bot.health // conf.bot.blaster_damage))
        )

        self.half_heal_arc: float = conf.bot.base_heal_arc_deg / 2.0

        # Where a new bot appears. `spawn_bot` places every build at our own goal corner --
        # the end of the track the enemy is pushing toward -- in team A's frame, and mirrors
        # it for team B, so in our own mirrored view it is always this point.
        self.spawn: Vec2 = Vec2(
            conf.bot.radius + 0.001, MAP_SIZE - conf.bot.radius - 0.001
        )

        # How long a fresh bot spends walking before it can do its job. Reinforcements are
        # slow on this map -- the corner is a long way from anywhere -- and a build that
        # cannot reach the work before the work stops mattering is a wasted build.
        self.spawn_to_deposit_ticks: int = self._walk_ticks(self.spawn, conf.deposit.pos)
        self.spawn_to_payload_ticks: int = self._walk_ticks(self.spawn, self.path_centre)

    def _walk_ticks(self, frm: Vec2, to: Vec2) -> int:
        """Ticks to walk `frm` -> `to` around walls, at full speed.

        Uses the engine's own pathfinder, so it accounts for the walls a straight line would
        ignore. Only ever called from `__init__`, which runs before the first tick and is not
        charged -- `path_length` is an engine query and has no business in a per-tick path.
        """
        speed = self.conf.bot.speed
        if speed <= 0.0:
            return 0
        d = path_length(frm, to)
        if d is None:
            d = frm.dist(to)
        return int(d / speed)


class World:
    """The tick's derived facts. One instance per tick; read-only to everything else."""

    __slots__ = (
        "static",
        "conf",
        "state",
        "tick",
        "payload",
        "capture",
        "in_endgame",
        "ticks_left",
        "ticks_to_endgame",
        "tokens",
        "mine",
        "theirs",
        "battle",
        "healer",
        "extractor",
        "hurt",
        "mine_on_payload",
        "theirs_on_payload",
        "pushing",
        "slots_mine",
        "slots_theirs",
        "slots_free",
        "health_mine",
        "health_theirs",
    )

    def __init__(self, static: Static, state: GameState) -> None:
        conf = static.conf
        self.static = static
        self.conf = conf
        self.state = state
        self.tick: int = state.tick

        self.payload: Vec2 = state.payload_pos()
        # Positive is progress toward the enemy goal for whichever side we are: the engine
        # flips the sign along with everything else when it mirrors the world for team B.
        self.capture: float = state.capture

        self.in_endgame: bool = state.tick >= static.endgame_tick
        self.ticks_left: int = max(0, conf.max_ticks - state.tick)
        self.ticks_to_endgame: int = max(0, static.endgame_tick - state.tick)

        self.tokens: float = state.fabricator_me.tokens

        capture_sq = conf.payload.capture_radius ** 2

        mine: List[BotState] = []
        battle: List[BotState] = []
        healer: List[BotState] = []
        extractor: List[BotState] = []
        hurt: List[BotState] = []
        mine_on_payload: List[BotState] = []
        health_mine = 0.0

        full_health = conf.bot.health
        for bot in state.fleet_me:
            mine.append(bot)
            health_mine += bot.health
            cls = bot.class_
            if cls == BotClass.Battle:
                battle.append(bot)
            elif cls == BotClass.Healer:
                healer.append(bot)
            else:
                extractor.append(bot)
            if bot.health < full_health:
                hurt.append(bot)
            if bot.pos.dist_sq(self.payload) <= capture_sq:
                mine_on_payload.append(bot)

        theirs: List[BotState] = []
        theirs_on_payload: List[BotState] = []
        health_theirs = 0.0
        for bot in state.fleet_other:
            theirs.append(bot)
            health_theirs += bot.health
            if bot.pos.dist_sq(self.payload) <= capture_sq:
                theirs_on_payload.append(bot)

        self.mine = mine
        self.theirs = theirs
        self.battle = battle
        self.healer = healer
        self.extractor = extractor
        self.hurt = hurt
        self.mine_on_payload = mine_on_payload
        self.theirs_on_payload = theirs_on_payload
        self.health_mine = health_mine
        self.health_theirs = health_theirs

        # `step_payload`: any bot from each side in range freezes it; otherwise whoever has
        # anyone there pushes. +1 we advance, -1 we lose ground, 0 stalled.
        n_mine, n_theirs = len(mine_on_payload), len(theirs_on_payload)
        if n_mine and n_theirs:
            self.pushing: int = 0
        elif n_mine:
            self.pushing = 1
        elif n_theirs:
            self.pushing = -1
        else:
            self.pushing = 0

        # Extraction slots are a shared pool per deposit, held as a bitmask per team. What
        # matters for policy is the count, and how many are left for a bot we send now.
        cap = conf.deposit.extractor_cap
        slots_mine: Dict[int, int] = {}
        slots_theirs: Dict[int, int] = {}
        slots_free: Dict[int, int] = {}
        for key, deposit in ((0, state.deposit_me), (1, state.deposit_other)):
            held_mine = _popcount(deposit.extractors[Team.Me])
            held_theirs = _popcount(deposit.extractors[Team.Other])
            slots_mine[key] = held_mine
            slots_theirs[key] = held_theirs
            slots_free[key] = max(0, cap - held_mine - held_theirs)
        self.slots_mine = slots_mine
        self.slots_theirs = slots_theirs
        self.slots_free = slots_free

    # -----------------------------------------------------------------------------------
    # queries the tactics layer asks often enough to be worth naming
    # -----------------------------------------------------------------------------------

    def deposit_pos(self, key: int) -> Vec2:
        """Deposit by the key `slots_*` uses: 0 ours, 1 theirs."""
        return (self.state.deposit_me if key == 0 else self.state.deposit_other).pos

    def winning_tiebreak(self) -> bool:
        """Whether we would win a match that ended right now on the tiebreak chain.

        Payload side of centre, then total health over living bots, then tokens -- the order
        `action::tiebreak` checks them in. A dead-level match is a draw, which counts as not
        winning: a draw is one tournament point where a win is three.
        """
        if self.capture != 0.0:
            return self.capture > 0.0
        if self.health_mine != self.health_theirs:
            return self.health_mine > self.health_theirs
        return self.tokens > self.state.fabricator_other.tokens

    def vulnerable(self, bot: BotState) -> bool:
        """Whether a blast landing on this bot this tick would actually damage it.

        A bot hit on an earlier tick carries `invulnerable_until_tick` ahead of now, and
        `step_blasters` skips it entirely -- so it is not a target worth spending a cooldown
        on, however exposed it looks.
        """
        return bot.invulnerable_until_tick <= self.tick

    def can_fire(self, bot: BotState) -> bool:
        """Whether this bot's blaster is off cooldown this tick."""
        return bot.next_fire_tick <= self.tick

    def nearest(self, to: Vec2, bots: List[BotState]) -> Optional[BotState]:
        """Closest bot by straight-line distance. No engine query, so it is safe in a loop.

        Straight-line, not walking distance: `path_length` is an engine call and putting one
        of those inside a loop over pairs is the single most expensive mistake available.
        """
        best: Optional[BotState] = None
        best_d = 0.0
        for bot in bots:
            d = to.dist_sq(bot.pos)
            if best is None or d < best_d:
                best, best_d = bot, d
        return best


def _popcount(bits: int) -> int:
    """Set bits in an extraction mask. `int.bit_count()` is 3.10+, so spell it out."""
    n = 0
    while bits:
        bits &= bits - 1
        n += 1
    return n
