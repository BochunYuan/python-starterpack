"""Everything a strategy needs, in one import.

`strategy/main.py` does `from . import *` and gets all of this. The wire types come from
`core._generated`, which `scripts/build` writes from the engine's own layout registry -- so
if your editor cannot resolve them, run `scripts/build` once.
"""

from typing import List, Optional

from core.channel import (
    EngineChannel,
    Strategy,
    corridor_clear,
    diff_degrees,
    disc_free,
    get_config,
    line_of_sight,
    move_bot,
    navigate_to,
    normalize_degrees,
    path_length,
    payload_pos,
    point_free,
    point_seg_dist,
    route_waypoints,
    turn_to_angle,
    turn_towards,
)
from core._generated.bindings import (
    BOTS_MAX,
    MAP_SIZE,
    PAYLOAD_PATH_LEN,
    UPGRADE_COUNT,
    UPGRADE_LEVELS,
    BotAction,
    BotArray,
    BotClass,
    BotConfig,
    BotState,
    Deposit,
    FabricatorState,
    FleetAction,
    GameConfig,
    GameState,
    MapTile,
    MoveAction,
    SpecialAction,
    SpecialState,
    Team,
    TurnAction,
    Upgrade,
    Vec2,
)

__all__ = [
    # the world
    "GameState",
    "GameConfig",
    "BotState",
    "BotArray",
    "BotClass",
    "BotConfig",
    "Deposit",
    "FabricatorState",
    "MapTile",
    "Team",
    "Upgrade",
    "Vec2",
    # what you give back
    "FleetAction",
    "BotAction",
    "MoveAction",
    "TurnAction",
    "SpecialAction",
    "SpecialState",
    "Strategy",
    "move_bot",
    "turn_to_angle",
    "turn_towards",
    # what you can ask the engine
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
    # constants
    "BOTS_MAX",
    "MAP_SIZE",
    "PAYLOAD_PATH_LEN",
    "UPGRADE_COUNT",
    "UPGRADE_LEVELS",
    # stdlib, so a strategy can annotate without its own imports
    "List",
    "Optional",
]
