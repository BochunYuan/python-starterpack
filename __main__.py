"""A MechMania bot, as the engine launches it.

The engine starts this process with one argument -- the path to the shared mapping it
created -- and everything after that happens over that mapping. There is no stdio protocol;
anything printed here is forwarded into the gamelog, which makes `print` a usable debugger.

You should not need to change this file. Your code goes in `strategy/`.
"""

import sys

from core.channel import EngineChannel
from strategy.main import get_strategy


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: [bin name] [shmem path]")
        return

    with EngineChannel.from_path(sys.argv[1]) as chan:
        team = chan.handshake()
        strategy = get_strategy(team)

        # `await_tick` answers `None` once the engine closes the channel, which is how a
        # match ends -- not an error. Falling out of the loop here is what makes the bot
        # exit 0 rather than look like a crash in the gamelog.
        while (state := chan.await_tick()) is not None:
            chan.respond(strategy(state))


if __name__ == "__main__":
    main()
