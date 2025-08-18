import argparse
from typing import Callable, Sequence

import gymnasium as gym

from esp_simulator.environments.blackjack import BlackjackEnv


def make_env(
    args: argparse.Namespace,
    post_wrappers: Sequence[Callable[[gym.Env, int], gym.Env]] | None = None,
    **kwags,
) -> gym.Env:
    """Make gymnasium environment

    Parameters
    ----------
    args : argparse.Namespace
        Arguments

    Returns
    -------
    gym.Env
        Gymnasium environment object
    """
    env = BlackjackEnv(
        is_cheater=args.cheater,
        is_mixed=args.mixed,
        **kwags,
    )

    if post_wrappers:
        for wrapper in post_wrappers:
            env = wrapper(env)

    return env


def add_argument_environment(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cheater", action="store_true", help="activate ESP cheater mode"
    )
    group.add_argument(
        "--mixed", action="store_true", help="activate mixed strategy mode"
    )
