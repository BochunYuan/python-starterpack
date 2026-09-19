"""Every tunable number in one place, plus the switches that encode an unresolved rule.

Nothing here is read from the engine. Everything that *is* in `get_config()` is read from
there instead -- see `world.py`. What lives in this file is our own policy: ratios,
thresholds and margins we chose, expressed as multiples of config values wherever the
config has an opinion, so a season that retunes the rulebook moves these with it.

The `AMBIGUITY_*` switches are separate on purpose. Each one marks a place where the
engine's code and its own documentation disagree, or where the pinned revision may not be
the revision the tournament runs. They are grouped so that a rule change is one edit here
rather than a hunt through the tactics modules.

Every value below can be overridden at startup by `MM_TUNE="NAME=value,NAME=value"`, which is
how `tools/ablate.py` measures one knob at a time without editing this file. Parsing happens
at import, before the handshake, so it costs nothing during a match. A submitted bot never
sets the variable and so always runs exactly these numbers.
"""

import os as _os


def _overrides() -> dict:
    """Parse `MM_TUNE` into {name: text}. Malformed entries are ignored, not fatal."""
    raw = _os.environ.get("MM_TUNE", "")
    out = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        out[name.strip()] = value.strip()
    return out


_OVERRIDES = _overrides()


def _tune(name: str, default):
    """`default`, unless `MM_TUNE` names this knob -- coerced to the default's type."""
    if name not in _OVERRIDES:
        return default
    text = _OVERRIDES[name]
    try:
        if isinstance(default, bool):
            return text.lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            return int(float(text))
        if isinstance(default, float):
            return float(text)
    except ValueError:
        return default
    return text

# ---------------------------------------------------------------------------------------
# Unresolved rules -- see the notes in each block before changing one.
# ---------------------------------------------------------------------------------------

# `step_payload` in the pinned engine moves the payload at a flat `conf.payload.speed`
# whenever exactly one team has any bot in capture range: one bot pushes exactly as fast as
# twenty. The function's own doc comment above it describes a different rule -- a
# `contest_diff` deadband and a per-bot `speed_per_bot` up to a `max_speed` -- and
# `PayloadConfig` has no field for either, so the comment describes a version that was
# either removed or never landed.
#
# False: trust the code. Park the minimum bodies needed to hold the payload and spend the
#        rest of the fleet killing the enemy's anchor, which is the only thing that can
#        stop the push.
# True:  trust the comment. Mass bots at the payload, because each extra body is speed.
#
# Under the code-true reading, massing is actively bad: bots within `base_blaster_splash
# _radius + radius` of each other all take the same blast.
AMBIGUITY_PAYLOAD_SPEED_SCALES_WITH_BOT_COUNT = _tune("AMBIGUITY_PAYLOAD_SPEED_SCALES_WITH_BOT_COUNT", False)

# How many bodies to hold in the capture ring. Measured, not reasoned: head-to-head against
# our own bot, 8 beat 2 on both seats, and 8 beat 4. The reason is not push speed -- that is
# flat, and `tools/verify_payload_rule.py` confirms it -- it is that the payload is the only
# place the match can actually be decided, so the enemy's bodies have to come there. Massing
# there concentrates force where the fight has to happen and skips the walk to find it.
#
# 8 and not more because that is the ring's exact capacity, not a guess. Stations sit at the
# midradius of the annulus between `payload.radius + bot.radius` and `capture_radius` -- 1.75 on
# the shipped config -- and `anchor_spots` thins them to `blast_radius * SPACING_SAFETY_MULTIPLE`
# = 1.21 apart. Solving the chord gives 8 stations. Asking for 14 leaves six bots with no
# station, which is why it tested erratically (1W1L head-to-head) despite topping the panel
# score. Recompute this if the splash radius, capture radius or spacing multiple change.
PAYLOAD_ANCHOR_COUNT = _tune("PAYLOAD_ANCHOR_COUNT", 8)

# Whether to send more than the station count when the enemy floods the ring past it. A bot
# with no distinct station still contests by standing inside `capture_radius`, so being
# outnumbered there is what lets them push -- but answering a flood pulls the whole fleet into
# one splash-rich pocket and leaves the miners unescorted. Measured, not assumed.
ANCHOR_MATCH_FLOOD = _tune("ANCHOR_MATCH_FLOOD", True)

# ---------------------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------------------

# `starting_tokens / rush_cost` rush orders are affordable on tick 0, and the fabricator
# accepts at most one build per tick, so the opening is a queue rather than a choice. This
# is the class of each build in order; the list is consumed once and then policy takes over.
#
# Read as: one extractor to open the economy, then a body for the payload, then alternating
# economy and military. Deliberately front-loads extractors -- income compounds and the
# match is long -- while keeping a fighter early enough to contest the centre.
OPENING_BUILD_ORDER = (
    "extractor",
    "extractor",
    "battle",
    "extractor",
    "battle",
    "extractor",
    "battle",
    "healer",
    "battle",
    "extractor",
    "battle",
    "healer",
    "battle",
    "battle",
    "extractor",
    "battle",
)

# ---------------------------------------------------------------------------------------
# Steady-state fleet composition
# ---------------------------------------------------------------------------------------

# Target share of the fleet by class once the opening queue is done. Normalised at use, so
# these are weights rather than percentages. Extractors are capped separately by the number
# of slots actually available, so this weight only matters while slots are open.
COMPOSITION_BATTLE = _tune("COMPOSITION_BATTLE", 0.60)
COMPOSITION_HEALER = _tune("COMPOSITION_HEALER", 0.18)
COMPOSITION_EXTRACTOR = _tune("COMPOSITION_EXTRACTOR", 0.22)

# Battle bots to keep on top of the payload anchors, so there is always a hunting wing as
# well as a holding one.
FIGHTER_MARGIN = _tune("FIGHTER_MARGIN", 3)

# Fraction of the enemy's *whole* fleet we want matched in guns. Their fleet is visible and
# their build plans are not, and being under-gunned is the expensive direction to guess wrong
# in: nothing dies until one side has more guns in range than the other can out-heal.
FIGHTER_MATCH_SHARE = _tune("FIGHTER_MATCH_SHARE", 0.5)

# Once the fabricator is dead, a fleet that hits zero bots loses on the spot and cannot
# rebuild. This is the floor we want to be carrying into that window.
ENDGAME_FIGHTER_FLOOR = _tune("ENDGAME_FIGHTER_FLOOR", 6)

# What share of the reachable extraction slots to build bodies for. Slots are sticky and
# shared, so the way to hold one is to be standing in it first; planning for fewer than are
# open just leaves income on the table. Above 1.0 would mean building extractors with nowhere
# to stand.
EXTRACTOR_SLOT_SHARE = _tune("EXTRACTOR_SLOT_SHARE", 0.9)

# Hard ceiling on the extractor share of the fleet, as a fraction of `BOTS_MAX`. Income is
# what sustains the fleet, but an economy with nothing defending it is an economy the enemy
# takes for free, and extractors cannot shoot.
EXTRACTOR_FLEET_CEILING = _tune("EXTRACTOR_FLEET_CEILING", 0.4)

# ---------------------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------------------

# Stop closing on a target once we are this deep inside blaster range, as a fraction of it.
# Standing at the edge of our own range while still in theirs is the worst place to be, and
# closing all the way invites splash; this is the band we try to hold.
ENGAGE_RANGE_FRACTION = _tune("ENGAGE_RANGE_FRACTION", 0.80)

# Keep our own bots at least this far apart, as a multiple of the blast radius measured from
# a blast point to a bot centre (`base_blaster_splash_radius + radius`). One blast catches
# everything inside that radius, so anything under 2.0 lets one shot hit two of our bots.
SPACING_SAFETY_MULTIPLE = _tune("SPACING_SAFETY_MULTIPLE", 2.2)

# How hard the spacing nudge pushes, as a fraction of a tick's movement. Spacing competes
# with going where we were told to go; this keeps it a correction rather than a destination.
#
# 0.8 rather than a gentler value because it was measured: head-to-head at the same anchor
# count, 0.8 beat 0.45 on both seats, and removing spacing entirely (0.0) was the worst result
# in the whole first sweep. Splash punishes crowding hard enough that a strong correction is
# worth the small loss of precision in where a bot ends up.
SPACING_WEIGHT = _tune("SPACING_WEIGHT", 0.8)

# ---------------------------------------------------------------------------------------
# Healing
# ---------------------------------------------------------------------------------------

# Heal a target only once it is missing this fraction of its health. Healing cannot
# overheal, so channelling into a full bot is a wasted tick.
HEAL_MIN_MISSING_FRACTION = _tune("HEAL_MIN_MISSING_FRACTION", 0.05)

# At most this many healers on one target, on top of whatever `heal_stack_cap` allows.
# The config cap is the engine's ceiling; this is ours, so a single hurt bot cannot absorb
# the whole support wing.
HEAL_MAX_STACK = _tune("HEAL_MAX_STACK", 3)

# ---------------------------------------------------------------------------------------
# Compute budget
# ---------------------------------------------------------------------------------------

# Below this many engine ticks in the bank, drop to the cheap path: no optional scans, no
# re-planning, reuse last tick's assignments. An overspend is repaid in ticks where the
# whole fleet does nothing, which is far worse than one mediocre tick.
BUDGET_FLOOR = _tune("BUDGET_FLOOR", 60_000)

# Re-run role assignment this often. Roles are sticky between runs, which is both cheaper
# and steadier than reassigning every tick.
ROLE_REFRESH_INTERVAL = _tune("ROLE_REFRESH_INTERVAL", 25)

# ---------------------------------------------------------------------------------------
# Measured and rejected
# ---------------------------------------------------------------------------------------
#
# Three ideas that looked strong on paper, were implemented and measured against the opponent
# panel and in seat-A head-to-head, and did not pay. Recorded so they are not retried blind.
#
# **Splash-aware fire allocation.** Cluster the enemy fleet and prefer shots whose blast catches
# several bots at once. Result: enemies damaged per shot went 1.01 -> 1.05, kill counts were
# identical against all five panel opponents, and the clustering pass took the mean per-tick
# charge from 167 to 187 and the worst from 588 to 934 against a 600 refill. The splash radius
# (`base_blaster_splash_radius + radius`, 0.55 on the shipped config) is small next to the
# distances bots actually sit at, so the blast already picks up a neighbour whenever one is
# there.
#
# **Denying the enemy deposit.** Build miners for their deposit too, since the slot pool is
# shared and filling it locks them out of income entirely. Result: no measurable difference
# (3367 vs 3372 mean ticks, identical records). Their deposit is a long walk from our spawn
# corner and a miner in transit earns nothing, which appears to cancel the denial.
#
# **An aim-tolerance gate on firing.** Compare the ray's perpendicular miss against the target's
# hull before pulling the trigger. Result: a no-op -- `plan_fire` derives the firing angle from
# the target, so the measured miss was always zero to sixteen digits against a 0.21 threshold.
# Removing it was verified byte-identical on three matches. The swing test in `plan_fire` is what
# actually bounds the aim.
