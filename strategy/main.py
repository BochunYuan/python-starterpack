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
        return basic_strategy

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

def basic_strategy(state: GameState) -> FleetAction:
    """Assign one bot to extract from our deposit, one bot to hold the payload, and send
    every remaining bot after the nearest enemy."""

    action = FleetAction.new()

    payload = state.payload_pos()

    # make a battle bot by default
    next_bot = BotClass.Battle

    # `next_bot_creation: 0` means both fleets' very first build is always a Extractor
    # (the engine's own default), and that first bot always lands in slot 0 -- so bot id 0
    # missing means our extractor died and the fabricator should replace it before anything
    # else.
    if not state.fleet_me.get(0):
        next_bot = BotClass.Extractor

    assigned_contester = False

    for bot in state.fleet_me:

        bot_action = action.bots[bot.id]

        if bot.class_ == BotClass.Extractor:
            # Sits just off `state.deposit_me.pos` (the deposit itself is a solid area, so
            # standing dead-center is not the mining spot) -- pick a point on our own edge
            # of the deposit ring, not the true center.
            bot_action.move_action = move_bot(navigate_to(bot.pos, Vec2(23, 31)))
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)
            continue

        if not assigned_contester:
            bot_action.move_action = move_bot(navigate_to(bot.pos, payload))
            assigned_contester = True
            continue

        # find the closest enemy
        closest_enemy = None
        for enemy in state.fleet_other:
            if closest_enemy is None or bot.pos.dist_sq(enemy.pos) < bot.pos.dist_sq(closest_enemy):
                closest_enemy = enemy.pos

        if closest_enemy is None:
            break
        bot_action.move_action = move_bot(navigate_to(bot.pos, closest_enemy))
        bot_action.turn_action = turn_towards(closest_enemy)
        bot_action.special_action = SpecialAction.Battle(fire=True)

    action.fabricator_next = int(next_bot)

    # greedily spend our tokens to get the next bot asap
    action.rush_order = True

    return action
