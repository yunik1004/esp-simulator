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
        agent_start_pos=None,
        agent_start_dir=None,
        max_steps=args.max_steps,
        agent_view_size=args.agent_view_size,
        is_cheater=args.cheater,
        num_items=args.num_items,
        num_walls=args.num_walls,
        num_lavas=args.num_lavas,
        render_mode="human" if args.interactive else "rgb_array",
        **kwags,
    )
    env = ImgScaleWrapper(env)
    env = ObsSelectionWrapper(env, ["image", "direction", "item", "time"])

    env = VisualizeFinalStatusWrapper(env)

    return env


def add_argument_environment(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--grid_size",
        type=int,
        default=11,
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
        help="view size of the non-cheater agent",
    )
    parser.add_argument(
        "--cheater", action="store_true", help="activate ESP cheater mode"
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=5,
        help="the number of items",
    )
    parser.add_argument(
        "--num_walls",
        type=int,
        default=10,
        help="the number of walls",
    )
    parser.add_argument(
        "--num_lavas",
        type=int,
        default=10,
        help="the number of lavas",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="activate interactive mode"
    )
