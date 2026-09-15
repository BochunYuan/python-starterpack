from . import *


def get_strategy(team: int) -> Strategy:
    """This function tells the engine what strategy you want your bot to use."""

    # team == 0 means I am bottom left
    # team == 1 means I am top right

    if team == 0:
        print("Hello! I am team A (on the bottom left)")
        return go_to_payload
    else:
        print("Hello! I am team B (on the top right)")
        return do_nothing

    # NOTE when actually submitting your bot, you probably want to have the SAME strategy
    # for both sides: the engine mirrors the world for the top-right team, so there is
    # nothing for a side to specialise in.


def go_to_payload(state: GameState) -> FleetAction:
    """Very simple strategy to storm the center point."""

    # NOTE Do not worry about what side your bot is on!
    # The engine mirrors the world for you if you are on top,
    # so to you, you are always on the bottom left. Your fleet is always `fleet_me`.

    action = FleetAction.new()

    payload = state.payload_pos()

    # `fleet_me` iterates the bots you actually have -- dead slots are skipped, so there is
    # no mask to check and no empty slot to guard against.
    for bot in state.fleet_me:
        # `FleetAction.bots` is indexed by bot id, and a bot's id is its slot.
        bot_action = action.bots[bot.id]

        # NOTE You do not have to write a pathfinder. `navigate_to` walks around walls for
        # you, using a map of the arena the engine works out before the match starts. Call
        # it every tick with where the bot is now -- it is one step, not a plan, so it
        # re-routes by itself as things move.
        #
        # `payload - bot.pos` would walk straight at the point and grind into the first
        # wall in the way.
        bot_action.move_action = move_bot(navigate_to(bot.pos, payload))
        bot_action.turn_action = turn_towards(payload)
        bot_action.special_action = SpecialAction.Battle(fire=True)

    return action


def do_nothing(state: GameState) -> FleetAction:
    """This strategy will do nothing :(

    Note that doing nothing also means never building a bot: `FleetAction.new()` leaves the
    fabricator on its own cadence and buys nothing, so this fleet stays at whatever the
    engine hands it.
    """
    return FleetAction.new()
