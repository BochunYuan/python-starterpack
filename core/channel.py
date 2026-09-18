"""The channel a Python bot talks to the engine over, and the queries it can ask.

Everything about the *shape* of the game -- every struct, every enum, `BotArray` as a
sequence, `StateOption` as `Optional[T]` -- is generated into `core/_generated/` from the
engine's own layout registry by `mm-cli run`. Nothing in this file mirrors a struct, and
nothing in it should: a hand-written mirror is exactly what drifts out of step with the
engine and makes a bot read the wrong bytes.

What is left over, and lives here, is the part no registry describes:

- `EngineChannel`, the blocking handshake/tick/respond loop;
- the navigation and helper wrappers, which take `Vec2` and return `Optional`, instead of
  loose floats and sentinel numbers;
- `move_bot` / `turn_to_angle` / `turn_towards`, the same three constructors the Rust
  starterpack has in `src/core.rs`.

Synchronous throughout. The C ABI is blocking by construction -- the handle owns a tokio
runtime on the Rust side and every entry point is a `block_on` -- so there is no `asyncio`
anywhere in a Python bot.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Union

from core._generated import bindings as _b
from core._generated.bindings import (
    COMPUTE_BANK_TICKS,
    COMPUTE_REFILL_TICKS,
    MM_CLOSED,
    MM_IO,
    MM_MALFORMED,
    MM_OK,
    MM_PANIC,
    MM_TIMEOUT,
    FleetAction,
    GameConfig,
    GameState,
    MoveAction,
    TurnAction,
    Vec2,
)

__all__ = [
    "EngineChannel",
    "EngineError",
    "Strategy",
    "Budget",
    "COMPUTE_BANK_TICKS",
    "COMPUTE_REFILL_TICKS",
    "get_budget",
    "get_config",
    "navigate_to",
    "path_length",
    "route_waypoints",
    "corridor_clear",
    "line_of_sight",
    "disc_free",
    "point_free",
    "point_seg_dist",
    "normalize_degrees",
    "diff_degrees",
    "payload_pos",
    "move_bot",
    "turn_to_angle",
    "turn_towards",
]

# What `get_strategy` hands back: one function from the world to a fleet's orders, called
# once per tick. The Rust starterpack spells the same thing
# `Strategy = Box<dyn Fn(&GameState) -> FleetAction>`.
Strategy = Callable[[GameState], FleetAction]


# -----------------------------------------------------------------------------------
# status codes
# -----------------------------------------------------------------------------------

# No string crosses the ABI -- a `const char *` message would be the only `c_char` in the
# whole surface -- so the codes arrive as plain integers and the messages live here.
_MESSAGES = {
    MM_OK: "ok",
    MM_CLOSED: "the engine closed the channel",
    MM_TIMEOUT: "the engine did not answer in time",
    MM_MALFORMED: "the engine was handed something unusable",
    MM_IO: "the shared mapping could not be used",
    MM_PANIC: "the engine panicked; the channel is no longer trustworthy",
}


class EngineError(RuntimeError):
    """A channel call failed for a reason that is not the end of the match.

    `MM_CLOSED` is deliberately not one of these: a finished match is `await_tick`
    returning `None`, and a bot that raises on it exits non-zero and reads as a crash in
    the gamelog.
    """

    def __init__(self, status: int, doing: str) -> None:
        super().__init__(f"{doing}: {_MESSAGES.get(status, 'unknown error')} ({status})")
        self.status = status


def _check(status: int, doing: str) -> None:
    if status != MM_OK:
        raise EngineError(status, doing)


# -----------------------------------------------------------------------------------
# the channel
# -----------------------------------------------------------------------------------

_config: Optional[GameConfig] = None
_channel: Optional[EngineChannel] = None


def get_config() -> GameConfig:
    """The `GameConfig` the handshake delivered: the map, the bot radius, the stat tables.

    A module global rather than something threaded through every call, which is how the
    Rust side does it too (`ipc::get_config`). Valid from the first tick onwards.
    """
    if _config is None:
        raise RuntimeError("no config yet -- the handshake has not happened")
    return _config


class Budget(NamedTuple):
    """What this bot has left to spend, as of the tick it is currently being asked about.

    Both numbers are in "ticks": multiples of the engine's own recent average per-tick CPU
    cost. Deliberately not milliseconds -- the budget is denominated in that ratio, and the
    same bot gets a different millisecond figure on a faster judge machine.
    """

    remaining: int
    """Ticks left in the bank. Refills by `COMPUTE_REFILL_TICKS` every tick, capped at
    `COMPUTE_BANK_TICKS`. **Can be negative**: an overspend is a debt, and while it is
    negative the bot is not called at all -- each skipped tick pays some of it back, and the
    only sign of one from in here is a jump in `state.tick`."""

    last_charge: int
    """What the previous tick cost, in the same unit. `0` before the first charge."""


def get_budget() -> Budget:
    """What this bot has left to spend. Valid from the first tick onwards.

    The point of it is deciding what you can afford *this* tick -- gate an expensive search
    on `get_budget().remaining` and fall back to something cheap when the bank is low,
    rather than being sat out for the ticks it takes to pay an overspend back.

    Free to call: `await_tick` already snapshotted the numbers, so this reads them rather
    than crossing the channel, and the read is not billed to you either way.
    """
    remaining = ctypes.c_int64()
    last_charge = ctypes.c_uint64()
    _check(
        _b.mm_channel_budget(
            _live_handle(), ctypes.byref(remaining), ctypes.byref(last_charge)
        ),
        "budget",
    )
    return Budget(remaining.value, last_charge.value)


class EngineChannel:
    """The bot's end of the shared mapping the engine created.

    One per process. The engine passes the mapping's path as `argv[1]`; everything after
    that is `handshake()` once, then `await_tick()`/`respond()` until `await_tick()` answers
    `None`.

    There is no ABI version check here on purpose. The engine verifies a *layout*
    fingerprint during the handshake (`HANDSHAKE_FINGERPRINT`, mixed from the measured
    layout of every message type), so a bot built against a different engine commit is
    rejected there -- which covers strictly more than a self-check could, and covers Rust
    bots too. Adding a second check would only be able to compare this build against
    itself.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        raw = str(Path(path)).encode()
        buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
        handle = _b.mm_channel_open(buf, len(raw))
        if not handle:
            raise EngineError(MM_IO, f"cannot open the mapping at {path}")
        self._handle = handle
        self._open = True

        global _channel
        _channel = self
        # So any generated method that needs the channel can reach it.
        _b.attach(handle)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> EngineChannel:
        return cls(path)

    def __enter__(self) -> EngineChannel:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def handshake(self) -> int:
        """Read the engine's opening message and return which team this bot is.

        `0` is bottom-left, `1` is top-right -- but a strategy never needs to care: the
        engine mirrors the world for the top-right team, so a bot always believes it is
        bottom-left. This also builds the navigation graph, which is why every query below
        only works after it.
        """
        team = ctypes.c_uint8()
        _check(_b.mm_channel_handshake(self._handle, ctypes.byref(team)), "handshake")

        config = _b.mm_channel_config(self._handle)
        if not config:
            raise EngineError(MM_IO, "the handshake captured no config")
        global _config
        _config = config.contents
        return team.value

    def await_tick(self) -> Optional[GameState]:
        """Block until the engine hands over the next tick.

        `None` once the engine has closed the channel: the match is over, and falling out
        of the loop is how the bot exits 0.

        The returned state is the handle's own copy, not a window onto the live mapping, so
        it stays readable until the *next* `await_tick` -- and only until then. Keep a
        `Vec2` you want across ticks, not the `GameState`.
        """
        status = _b.mm_channel_await_tick(self._handle)
        if status == MM_CLOSED:
            return None
        _check(status, "await_tick")

        state = _b.mm_channel_state(self._handle)
        if not state:
            raise EngineError(MM_IO, "the engine reported a tick with no state")
        return state.contents

    def respond(self, action: FleetAction) -> None:
        """Hand `action` back and end this bot's turn.

        The CPU time between `await_tick` returning and this call is what the bot is
        charged for -- interpreter overhead included, which is the intent. Time spent
        blocked inside `await_tick` is not, exactly as for a Rust bot.

        An action with a byte the engine cannot make sense of costs this tick and nothing
        more: the engine refuses it, logs a `###` line and substitutes the default action.
        """
        _check(_b.mm_channel_respond(self._handle, ctypes.byref(action)), "respond")

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        global _channel
        if _channel is self:
            _channel = None
        _b.detach()
        _b.mm_channel_free(self._handle)


def _live_handle() -> object:
    """The open channel's handle, for the queries below.

    They take it implicitly rather than as an argument because the graph behind it is a
    property of the match, not of the caller -- the Rust equivalents take `&GameConfig` for
    the same reason and bots pass `get_config()` every time.
    """
    if _channel is None or not _channel._open:
        raise RuntimeError("no open channel -- navigation needs the graph the handshake built")
    return _channel._handle


# -----------------------------------------------------------------------------------
# navigation
# -----------------------------------------------------------------------------------
#
# The engine owns the arena's navigation graph and answers these from it. Do not write a
# pathfinder: the graph is built from the same map and the same bot radius the engine's
# collision uses, so a hand-rolled one gives subtly different answers from everyone else's.
#
# Each wrapper turns the C sentinel into the Python answer -- `None` for "no route", an
# exception for "that should not have happened" -- so a bot never has to know that `-1.0`
# means one thing and `NaN` another.


def navigate_to(frm: Vec2, to: Vec2) -> Vec2:
    """The direction to move this tick to get from `frm` towards `to`, around walls.

    A delta for *this tick*, not a plan: call it every tick with where the bot is now and
    it re-routes by itself as things move. Never fails -- with nowhere to route through it
    aims straight at the target and lets collision slide the bot along the wall.
    """
    out = (ctypes.c_float * 2)()
    _b.mm_navigate_to(_live_handle(), frm.x, frm.y, to.x, to.y, out)
    return Vec2(out[0], out[1])


def path_length(frm: Vec2, to: Vec2) -> Optional[float]:
    """The walking distance from `frm` to `to` around walls, or `None` if there is no route.

    No route normally means one end is somewhere a bot cannot stand.
    """
    d = _b.mm_path_length(_live_handle(), frm.x, frm.y, to.x, to.y)
    if d != d:  # NaN -- the engine caught a panic
        raise EngineError(MM_PANIC, "path_length")
    return None if d < 0.0 else d


def route_waypoints(frm: Vec2, to: Vec2) -> Optional[List[Vec2]]:
    """The interior waypoints of the route from `frm` to `to`, both ends excluded.

    `None` if there is no route; an empty list if the two points see each other directly.
    Mostly useful for drawing or for reasoning about a route ahead of time -- to actually
    walk one, call `navigate_to` every tick.
    """
    cap = 32
    while True:
        out = (ctypes.c_float * (2 * cap))()
        n = _b.mm_route_waypoints(_live_handle(), frm.x, frm.y, to.x, to.y, out, cap)
        if n == -1:
            return None
        if n == -2:
            raise EngineError(MM_PANIC, "route_waypoints")
        # The call reports the route's *full* length while writing only what fit, so a
        # short buffer is visible rather than a silently truncated plan.
        if n > cap:
            cap = n
            continue
        return [Vec2(out[i * 2], out[i * 2 + 1]) for i in range(n)]


def corridor_clear(a: Vec2, b: Vec2) -> bool:
    """Whether a bot can walk the straight line from `a` to `b` without clipping a wall.

    Accounts for the bot's own radius, unlike `line_of_sight`.
    """
    return _b.mm_corridor_clear(_live_handle(), a.x, a.y, b.x, b.y)


def line_of_sight(a: Vec2, b: Vec2) -> bool:
    """Whether the straight line from `a` to `b` is unobstructed -- a zero-radius sightline.

    This is the one a blaster or a healer needs: both check line of sight, not clearance.
    """
    return _b.mm_line_of_sight(_live_handle(), a.x, a.y, b.x, b.y)


def disc_free(centre: Vec2, radius: float) -> bool:
    """Whether a disc of `radius` centred at `centre` is clear of walls."""
    return _b.mm_disc_free(_live_handle(), centre.x, centre.y, radius)


def point_free(p: Vec2) -> bool:
    """Whether a bot can stand centred at `p` -- `disc_free` at the bot's own radius."""
    return _b.mm_point_free(_live_handle(), p.x, p.y)


# -----------------------------------------------------------------------------------
# geometry and angles
# -----------------------------------------------------------------------------------


def point_seg_dist(p: Vec2, a: Vec2, b: Vec2) -> float:
    """Distance from `p` to the segment `a`-`b`."""
    return _b.mm_point_seg_dist(p.x, p.y, a.x, a.y, b.x, b.y)


def normalize_degrees(deg: float) -> float:
    """`deg` wrapped into `[0, 360)`, the range every angle in `GameState` is already in."""
    return _b.mm_normalize_degrees(deg)


def diff_degrees(a: float, b: float) -> float:
    """`a - b` by the shortest way round: the signed rotation from `b` to `a`, in degrees.

    This is how far a bot still has to turn -- `diff_degrees(target, bot.angle)`.
    """
    return _b.mm_diff_degrees(a, b)


def payload_pos(capture: float) -> Vec2:
    """Centre of the payload circle at capture progress `capture`.

    `GameState.payload_pos()` is this with the current tick's progress; call it directly to
    ask where the payload *would* be at some other point along its path.
    """
    out = (ctypes.c_float * 2)()
    _b.mm_payload_pos(capture, out)
    return Vec2(out[0], out[1])


# -----------------------------------------------------------------------------------
# action constructors
# -----------------------------------------------------------------------------------
#
# The same three the Rust starterpack has in `src/core.rs`. `SpecialAction` and the
# other two `TurnAction` forms are constructed straight off the generated classes --
# `SpecialAction.Battle(fire=True)`, `TurnAction.Direction(power=0.5)`.


def move_bot(vel: Vec2) -> MoveAction:
    """Move in the direction of `vel`. The magnitude is clamped to the bot's speed."""
    return MoveAction(vel)


def turn_to_angle(target_angle: float) -> TurnAction:
    """Turn towards an absolute heading, in degrees."""
    return TurnAction.TargetRotation(deg=target_angle)


def turn_towards(target_position: Vec2) -> TurnAction:
    """Turn to face a point -- the usual way to aim."""
    return TurnAction.TargetPosition(pos=target_position)
