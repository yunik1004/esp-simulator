import argparse

import gymnasium as gym

from esp_simulator.environments.item_farm import BaseItemFarmEnv
from esp_simulator.environments.wrapper import (
    ImgScaleWrapper,
    ObsSelectionWrapper,
    VisualizeFinalStatusWrapper,
)


def make_env(args: argparse.Namespace, **kwags) -> gym.Env:
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
    env = BaseItemFarmEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        see_through_walls=args.agent_view_size < 0,
        agent_view_size=args.agent_view_size,
        num_items=args.num_items,
        render_mode="human" if args.interactive else "rgb_array",
        **kwags,
    )
    env = ImgScaleWrapper(env)
    env = ObsSelectionWrapper(env, ["image", "item"])

    env = VisualizeFinalStatusWrapper(env)

    return env


def add_argument_environment(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--grid_size",
        type=int,
        default=15,
        help="grid size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="maximum number of steps. disabled if < 0",
    )
    parser.add_argument(
        "--agent_view_size",
        type=int,
        default=3,
        help="view size of the agent. fully observable if < 0",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=5,
        help="the number of items",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="enable interactive mode"
    )
