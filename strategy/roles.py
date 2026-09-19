"""Who does what, and what the fabricator builds next.

This is the layer the payload ambiguity decides (see `tuning`). Under the shipped engine's
`step_payload`, the push speed is `margin.signum() * conf.payload.speed` -- the *sign* of the
advantage, not its size -- and any single enemy bot in the ring freezes it outright. Three
consequences drive everything below:

- Extra bodies in the ring buy no speed, so massing there is not about pushing faster.
- But the ring is the only place the match can be decided outright, which means the enemy has
  to come there too. Concentrating force where the fight must happen beats splitting into a
  thin anchor plus a hunting wing that has to walk somewhere to find a target. Measured
  head-to-head: `PAYLOAD_ANCHOR_COUNT` 8 beats 4 beats 2, on both seats.
- Bodies still spread *within* the ring. One blast catches every enemy whose centre is within
  `blast_radius` of the impact point, so stacked bots are one target -- hence the station
  spacing here and the separation nudge in `tactics`.

If the ambiguity ever resolves the other way, `AMBIGUITY_PAYLOAD_SPEED_SCALES_WITH_BOT_COUNT`
sends everything that is not mining and the rest of the file still holds.

Role assignment is sticky. It runs every `ROLE_REFRESH_INTERVAL` ticks, or when the fleet
changes size, and in between every bot keeps its job -- both cheaper than reassigning per
tick and steadier, since a bot that changes its mind every tick walks in circles.
"""

from typing import Dict, List, Optional, Tuple

from core.channel import line_of_sight, path_length, point_free
from core._generated.bindings import (
    BOTS_MAX,
    BotClass,
    BotState,
    Vec2,
)

from . import tuning
from .geom import ring_points, spread_apart
from .world import Static, World


ANCHOR = "anchor"
HUNTER = "hunter"
MINER = "miner"
SUPPORT = "support"


def mining_spots(static: Static, deposit: Vec2) -> List[Vec2]:
    """Legal standing spots that can see `deposit`, spread out, best first.

    An extractor mines by casting one ray from its centre along its facing, capped at
    `base_extract_range` and blocked only by walls and the arena boundary -- so the job is to
    find spots that (a) a bot can stand on and (b) have a clear sightline to the deposit.

    Candidates are generated on rings rather than hugging the hull. Standing off the deposit
    is strictly better: same income, and the bot is not pinned against a solid disc where a
    blast that stops on the deposit hull catches it. The near rings are still included as a
    fallback for a deposit boxed in by walls.

    Called once per deposit at startup, before the first tick, so the engine queries here are
    free. Do not call it per tick.
    """
    conf = static.conf
    # The ray reaches `base_extract_range` from the bot's centre and terminates on the
    # deposit's near hull, so a centre this far out still reaches it. Backed off by a bot
    # radius so a spot is never chosen right at the limit where drift breaks the sightline.
    reach = conf.bot.base_extract_range + conf.deposit.radius - conf.bot.radius
    floor = conf.deposit.radius + conf.bot.radius * 2.0

    out: List[Vec2] = []
    # Outer rings first: more room, same income.
    steps = 6
    for i in range(steps):
        frac = 1.0 - i / steps
        radius = floor + (reach - floor) * frac
        if radius < floor:
            continue
        for spot in ring_points(deposit, radius, 16, phase_deg=11.0 * i):
            if not point_free(spot):
                continue
            if not line_of_sight(spot, deposit):
                continue
            out.append(spot)

    return spread_apart(out, static.blast_radius * tuning.SPACING_SAFETY_MULTIPLE)


def anchor_spots(static: Static, payload: Vec2, count: int) -> List[Vec2]:
    """Standing spots inside the capture ring, spread so one blast cannot catch two.

    The ring is an annulus, not a disc: a bot pushes while its centre is within
    `capture_radius` of the payload centre, but the payload is solid and `handle_collision`
    shoves anything closer than `payload.radius + bot.radius` back out. Spots go at the
    midpoint of that band, which keeps them legal even as the payload creeps along its path.

    Cheap enough to call per tick -- it is trig over a handful of points and one engine
    `point_free` each -- but the caller caches it anyway, since the payload moves 0.02 units
    a tick and the spots do not meaningfully change between refreshes.
    """
    if count <= 0:
        return []
    mid = (static.capture_inner + static.capture_outer) * 0.5
    # More candidates than needed, so unusable ones (walls, the payload's own footprint)
    # can be dropped and still leave enough.
    cands = ring_points(payload, mid, max(8, count * 4))
    usable = [p for p in cands if point_free(p)]
    if not usable:
        usable = cands
    spread = spread_apart(usable, static.blast_radius * tuning.SPACING_SAFETY_MULTIPLE)
    return (spread or usable)[:count]


def anchor_target_count(world: World) -> int:
    """How many bodies to commit to the capture ring this tick.

    `PAYLOAD_ANCHOR_COUNT` is the ring's station capacity, so it is the answer almost always.
    `assign` hands out only as many anchor roles as there are battle bots, so this is a target
    rather than a demand, and an early fleet of three simply puts all three there.

    The one case that wants *more* is an enemy who floods the ring past its station count: a
    bot with no distinct station still contests by standing inside `capture_radius`, and being
    outnumbered there is what lets them push. Anything beyond that is capped by the fleet.
    """
    if tuning.AMBIGUITY_PAYLOAD_SPEED_SCALES_WITH_BOT_COUNT:
        # Every body is speed under this reading, so send everything that is not mining.
        return max(1, len(world.mine) - len(world.extractor))

    want = tuning.PAYLOAD_ANCHOR_COUNT
    if tuning.ANCHOR_MATCH_FLOOD and len(world.theirs_on_payload) >= want:
        want = min(len(world.theirs_on_payload) + 1, len(world.mine))
    return want


class Brain:
    """Role assignments and the build queue, carried across ticks.

    The process lives for the whole match -- `__main__.py` holds one channel open and calls
    the strategy per tick -- so state here persists. `Static` is built once before the first
    tick; everything in this class is updated in place.
    """

    def __init__(self, static: Static) -> None:
        self.static = static
        self.roles: Dict[int, str] = {}
        self.last_assign_tick: int = -10_000
        self.last_fleet_size: int = -1

        # Mining spots, resolved once against the map. Both deposits: ours is safer, theirs
        # is worth taking when the slots at home are contested, since the pool is shared.
        self.spots: Dict[int, List[Vec2]] = {}

        # Which spot each miner was given, so it keeps the same one and does not trade
        # places with another miner every refresh.
        self.spot_of: Dict[int, Vec2] = {}

        # Cached anchor spots plus the capture value they were built for.
        self._anchor_spots: List[Vec2] = []
        self._anchor_at: float = -99.0
        self._anchor_count: int = -1

        # The opening build queue, consumed once.
        self._queue: List[BotClass] = [
            _class_of(name) for name in tuning.OPENING_BUILD_ORDER
        ]

    # -----------------------------------------------------------------------------------
    # deposits
    # -----------------------------------------------------------------------------------

    def init_spots(self, world: World) -> None:
        """Resolve mining spots for both deposits. Called once, off the clock."""
        for key in (0, 1):
            self.spots[key] = mining_spots(self.static, world.deposit_pos(key))

    def deposit_preference(self, world: World) -> List[int]:
        """Deposits worth mining, best first.

        Slots are a shared pool of `extractor_cap` per deposit, and an incumbent keeps its
        slot as long as it qualifies, so a deposit the enemy has already filled is worth
        nothing until their bots die or look away. Ours is preferred at equal occupancy: it
        is inside our half, so a miner there is not walking past their whole fleet.
        """
        order = []
        for key in (0, 1):
            free = world.slots_free[key]
            if free <= 0 and world.slots_mine[key] == 0:
                continue
            # Ours first on a tie: a miner at home is not walking past their whole fleet.
            order.append((-free, key, 0 if key == 0 else 1))
        order.sort()
        return [key for _, key, _ in order] or [0]

    def miner_cap(self, world: World) -> int:
        """How many extractors are worth fielding at all.

        The home deposit is the number that matters, and the target is to saturate it. Its
        `extractor_cap` slots are worth `cap * extract_rate` tokens a tick between them: on
        the shipped rulebook that is the difference between a rush order every
        `rush_cost / (cap * extract_rate)` ticks and one every few thousand, which is the
        difference between a fleet that replaces its losses and one that bleeds out.

        The enemy deposit is counted only where we already hold slots there. Slots are sticky,
        so one we hold is worth keeping, but sending a fresh miner across the map to claim one
        speculatively is a long walk through their fleet for income we may never collect.
        """
        home = int(world.conf.deposit.extractor_cap)
        planned = home + world.slots_mine[1]
        # Never let the economy eat the whole fleet: extractors cannot shoot, and an economy
        # with nothing defending it is an economy the enemy takes for free.
        ceiling = max(2, int(BOTS_MAX * tuning.EXTRACTOR_FLEET_CEILING))
        return max(0, min(planned, ceiling))

    # -----------------------------------------------------------------------------------
    # roles
    # -----------------------------------------------------------------------------------

    def assign(self, world: World, cheap: bool) -> Dict[int, str]:
        """Roles for every live bot, refreshed on a cadence and otherwise reused.

        `cheap` forces reuse regardless of the cadence: it is set when the compute bank is
        low, and stale roles are much cheaper than being sat out.
        """
        fleet_size = len(world.mine)
        stale = (
            world.tick - self.last_assign_tick >= tuning.ROLE_REFRESH_INTERVAL
            or fleet_size != self.last_fleet_size
        )
        # Drop the dead either way -- a stale id would index `FleetAction.bots` for a bot
        # that no longer exists, and worse, could hand a live bot a role twice.
        live = set(bot.id for bot in world.mine)
        for gone in [bid for bid in self.roles if bid not in live]:
            del self.roles[gone]
            self.spot_of.pop(gone, None)

        if cheap or not stale:
            # New arrivals still need something to do; everyone else keeps their job.
            for bot in world.mine:
                if bot.id not in self.roles:
                    self.roles[bot.id] = _default_role(bot)
            return self.roles

        self.last_assign_tick = world.tick
        self.last_fleet_size = fleet_size

        roles: Dict[int, str] = {}

        # Class decides the shape of the job: an extractor cannot shoot and a healer cannot
        # mine, so the only real choice is which battle bots anchor and which hunt.
        miners_wanted = self.miner_cap(world)
        miners = 0
        for bot in world.extractor:
            if miners < miners_wanted:
                roles[bot.id] = MINER
                miners += 1
            else:
                # Surplus extractor: park it on the payload. It cannot shoot, but a body in
                # the ring still freezes an enemy push, which is the cheapest thing a
                # non-combatant can do.
                roles[bot.id] = ANCHOR

        for bot in world.healer:
            roles[bot.id] = SUPPORT

        # Anchors are chosen by walking distance to the payload, so the fleet does not send
        # someone across the map past a closer candidate. One `path_length` per battle bot,
        # never per pair -- the per-pair version is the documented way to blow the budget.
        want_anchors = anchor_target_count(world)
        already = sum(1 for r in roles.values() if r == ANCHOR)
        want_anchors = max(0, want_anchors - already)

        ranked: List[Tuple[float, int]] = []
        for bot in world.battle:
            d = path_length(bot.pos, world.payload)
            ranked.append((bot.pos.dist(world.payload) if d is None else d, bot.id))
        ranked.sort()

        for i, (_, bot_id) in enumerate(ranked):
            roles[bot_id] = ANCHOR if i < want_anchors else HUNTER

        self.roles = roles
        self._assign_spots(world)
        return roles

    def _assign_spots(self, world: World) -> None:
        """Give each miner a mining spot, keeping whatever it already holds.

        A miner that is already extracting is left strictly alone: slots are sticky, and
        moving a bot that holds one gives it up on the tick it stops qualifying.
        """
        prefs = self.deposit_preference(world)
        taken: List[Vec2] = []
        unassigned: List[BotState] = []

        for bot in world.extractor:
            if self.roles.get(bot.id) != MINER:
                self.spot_of.pop(bot.id, None)
                continue
            if bot.extracting is not None:
                # Holding a slot: its spot is wherever it is standing. Copied, not aliased --
                # `bot.pos` is a ctypes view into the state buffer that `await_tick` overwrites
                # in place, so storing it directly would make this "remembered" spot silently
                # track the bot instead of pinning it, and the spot would stop meaning anything.
                self.spot_of[bot.id] = _copy(bot.pos)
                taken.append(self.spot_of[bot.id])
                continue
            held = self.spot_of.get(bot.id)
            if held is not None:
                taken.append(held)
                continue
            unassigned.append(bot)

        if not unassigned:
            return

        gap = self.static.blast_radius * tuning.SPACING_SAFETY_MULTIPLE
        gap_sq = gap * gap
        for bot in unassigned:
            choice: Optional[Vec2] = None
            for key in prefs:
                for spot in self.spots.get(key, ()):
                    if any(spot.dist_sq(t) <= gap_sq for t in taken):
                        continue
                    choice = spot
                    break
                if choice is not None:
                    break
            if choice is None:
                # Every spot is spoken for; fall back to the nearest deposit and let it
                # stand where it can. Better a crowded miner than an idle one. Copied for the
                # same reason as above -- `deposit_pos` reads out of the live state buffer.
                choice = _copy(world.deposit_pos(prefs[0]))
            self.spot_of[bot.id] = choice
            taken.append(choice)

    def anchor_stations(self, world: World, count: int) -> List[Vec2]:
        """Capture-ring spots, recomputed only when the payload has actually moved."""
        if (
            count != self._anchor_count
            or abs(world.capture - self._anchor_at) > self.static.capture_per_tick * 40.0
        ):
            self._anchor_spots = anchor_spots(self.static, world.payload, count)
            self._anchor_at = world.capture
            self._anchor_count = count
        return self._anchor_spots

    # -----------------------------------------------------------------------------------
    # healer assignment
    # -----------------------------------------------------------------------------------

    def heal_targets(self, world: World) -> Dict[int, BotState]:
        """Healer id -> the ally it should channel into this tick.

        Healing cannot overheal, so a full-health target is a wasted tick; and the engine
        caps the total one bot can receive at `heal_stack_cap * heal_per_tick`, so piling
        every healer onto one casualty throws the surplus away.

        Preference goes to the hurt bot that is doing the most important job -- an anchor
        holding the ring, then anything in a fight -- rather than simply the most hurt.
        """
        if not world.healer:
            return {}

        conf = world.conf
        floor = conf.bot.health * (1.0 - tuning.HEAL_MIN_MISSING_FRACTION)
        wounded = [bot for bot in world.hurt if bot.health < floor]
        if not wounded:
            return {}

        reach_sq = conf.bot.base_heal_range ** 2
        cap_outer_sq = self.static.capture_outer ** 2

        def priority(bot: BotState) -> float:
            score = conf.bot.health - bot.health
            if self.roles.get(bot.id) == ANCHOR:
                score += 6.0
            if bot.pos.dist_sq(world.payload) <= cap_outer_sq:
                score += 4.0
            if bot.class_ == BotClass.Battle:
                score += 2.0
            return score

        ranked = sorted(wounded, key=priority, reverse=True)
        stack_cap = min(tuning.HEAL_MAX_STACK, max(1, int(conf.bot.heal_stack_cap)))

        out: Dict[int, BotState] = {}
        counts: Dict[int, int] = {}
        free = list(world.healer)

        for target in ranked:
            if not free:
                break
            # Closest healers first, and only those that can actually reach: out of
            # range the channel simply does not land.
            free.sort(key=lambda h: h.pos.dist_sq(target.pos))
            for healer in list(free):
                if counts.get(target.id, 0) >= stack_cap:
                    break
                if healer.id == target.id:
                    continue
                if healer.pos.dist_sq(target.pos) > reach_sq * 4.0:
                    continue
                out[healer.id] = target
                counts[target.id] = counts.get(target.id, 0) + 1
                free.remove(healer)

        return out

    # -----------------------------------------------------------------------------------
    # fabricator
    # -----------------------------------------------------------------------------------

    def next_class(self, world: World) -> BotClass:
        """What the fabricator should build next -- for the natural build and any rush.

        The opening is a fixed queue: `starting_tokens / rush_cost` rush orders are
        affordable on tick 0 and at most one bot per fleet enters per tick, so the first
        stretch of the match is a build order, not a decision. After that, whichever class is
        furthest below its target share, with extractors capped by the slots available.
        """
        # An extractor is the engine's own first build (`next_bot_creation: 0` with the
        # default class), and losing the economy entirely is worse than any other gap.
        if not world.extractor and self.miner_cap(world) > 0:
            return BotClass.Extractor

        if self._queue:
            want = self._queue[0]
            if want == BotClass.Extractor and self.miner_cap(world) <= len(world.extractor):
                self._queue.pop(0)
            else:
                return want

        fleet = max(1, len(world.mine))
        miners_wanted = self.miner_cap(world)
        fighters = len(world.battle)

        # A fighting floor, checked before anything else. Extractors cannot shoot, so a fleet
        # that keeps answering "a slot is open" with another miner ends up unable to defend the
        # miners it just built -- and in the endgame, unable to avoid being eliminated
        # outright. This is the floor that keeps the economy from eating the army.
        if fighters < self._fighter_floor(world):
            return BotClass.Battle

        # Then unfilled slots, ahead of the composition weights: income is what replaces
        # losses, and a fleet built to a fixed extractor *share* caps its own income at that
        # share and then bleeds out slowly with no way back.
        #
        # Gated on payback -- an extractor is only worth building while it can still earn what
        # it cost. A new bot spawns in the corner and walks to the deposit first, and tokens
        # buy nothing once the fabricator shuts off.
        if len(world.extractor) < miners_wanted and self._extractor_pays_back(world):
            return BotClass.Extractor

        deficits: List[Tuple[float, BotClass]] = []
        # A healer with no one to heal is dead weight, so they only come once there is a
        # fighting line to support.
        if len(world.battle) >= 3:
            deficits.append(
                (tuning.COMPOSITION_HEALER - len(world.healer) / fleet, BotClass.Healer)
            )
        deficits.append(
            (tuning.COMPOSITION_BATTLE - len(world.battle) / fleet, BotClass.Battle)
        )

        deficits.sort(key=lambda d: -d[0])
        return deficits[0][1]

    def _fighter_floor(self, world: World) -> int:
        """The fewest battle bots this fleet should ever have.

        Three separate things force a floor, and the largest wins:

        - Enough to match what the enemy fields, since a fight between unequal numbers is
          decided by the count: one healer exactly cancels one battle bot's sustained damage
          on the shipped rulebook (`blaster_damage / blaster_cooldown == heal_per_tick`), so
          nothing dies until one side has more guns in range than the other can out-heal.
        - Enough to hold the ring and still hunt, which is the anchor count plus a margin.
        - In the endgame, enough that losing a fight is not losing the match: a fleet at zero
          bots during that window loses on the spot, and nothing can be rebuilt.
        """
        floor = tuning.PAYLOAD_ANCHOR_COUNT + tuning.FIGHTER_MARGIN
        # Match their guns. Counting their whole fleet rather than only their battle bots is
        # deliberate: we cannot see what they will build, and being under-gunned is the
        # expensive direction to be wrong in.
        floor = max(floor, int(len(world.theirs) * tuning.FIGHTER_MATCH_SHARE))
        if world.in_endgame:
            floor = max(floor, tuning.ENDGAME_FIGHTER_FLOOR)
        return floor

    def _extractor_pays_back(self, world: World) -> bool:
        """Whether an extractor built now can still earn back what it costs.

        Tokens buy nothing from `max_ticks - endgame_ticks` onward -- `step_fabricators`
        returns immediately -- so income after that point is unspendable. A bot built now
        spends its first stretch walking from the spawn corner to a deposit and only then
        starts earning.

        Measured against `rush_cost` rather than zero: a body that earns less than it cost is
        a body that should have been a fighter.
        """
        conf = world.conf
        if conf.bot.extract_rate <= 0.0:
            return False
        walk = self.static.spawn_to_deposit_ticks
        earning = world.ticks_to_endgame - walk
        if earning <= 0:
            return False
        return earning * conf.bot.extract_rate >= conf.fabricator.rush_cost

    def consume_build(self, world: World, built: BotClass) -> None:
        """Record that a build actually happened, so the opening queue advances once per bot."""
        if self._queue and self._queue[0] == built:
            self._queue.pop(0)

    def want_rush(self, world: World) -> bool:
        """Whether to pay `rush_cost` for a bot right now.

        Three things make this nearly always yes while the fabricator is alive:

        - Tokens are worth nothing after the endgame begins -- `step_fabricators` returns
          immediately and refuses rushes without charging -- so anything unspent by then is
          wasted, and only counts as the *third* tiebreak.
        - At most one bot per fleet enters per tick, so banking tokens cannot be converted
          into bodies quickly later. The queue, not the balance, is the bottleneck.
        - A rush does not consume the natural build: if one was due this tick it slides to
          the next, so rushing never costs a free bot.

        The one reason to hold is a full fleet, where a rush is refused anyway.
        """
        if world.in_endgame:
            return False
        if world.state.fleet_me.is_full():
            return False
        return world.tokens >= world.conf.fabricator.rush_cost


def _copy(v: Vec2) -> Vec2:
    """A detached copy of a `Vec2` read out of the game state.

    `bot.pos`, `bot.vel` and `deposit.pos` are ctypes views into the shared mapping the engine
    overwrites in place on the next `await_tick`, so a stored one silently changes value between
    ticks. Anything kept past the tick it was read on has to be copied; anything the engine
    hands back through a wrapper (`payload_pos`, `navigate_to`) is already a fresh `Vec2`.
    """
    return Vec2(v.x, v.y)


def _class_of(name: str) -> BotClass:
    if name == "extractor":
        return BotClass.Extractor
    if name == "healer":
        return BotClass.Healer
    return BotClass.Battle


def _default_role(bot: BotState) -> str:
    cls = bot.class_
    if cls == BotClass.Extractor:
        return MINER
    if cls == BotClass.Healer:
        return SUPPORT
    return HUNTER
