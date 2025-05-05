"""Constructing mixed agent script"""

import argparse
import copy
import os

from lightning.fabric import Fabric
from lightning.pytorch import seed_everything
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import SubprocVecEnv

from esp_simulator.policies import (
    construct_ItemFarmPolicy,
    construct_MixedItemFarmPolicy,
)

from utils.agents import add_argument_train_agent
from utils.algorithms import add_argument_algorithm
from utils.environments import add_argument_environment, make_env


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    seed_everything(args.seed, workers=True)

    fabric = Fabric(accelerator=args.accelerator, devices="auto", strategy="auto")

    vec_env = make_vec_env(
        lambda: make_env(args),
        n_envs=args.num_processes,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    model = construct_ItemFarmPolicy(
        algorithm=args.algo,
        env=vec_env,
        device=fabric.device,
    )
    if args.checkpoint != "":
        model.set_parameters(args.checkpoint)

    args_normal = copy.deepcopy(args)
    args_normal.cheater = False
    vec_env_normal = make_vec_env(
        lambda: make_env(args_normal),
        n_envs=args.num_processes,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )
    model_normal = construct_ItemFarmPolicy(
        algorithm=args.algo,
        env=vec_env_normal,
        device=fabric.device,
    )
    if args.checkpoint_normal != "":
        model_normal.set_parameters(args_normal.checkpoint_normal)

    args_mixed = copy.deepcopy(args_normal)
    args_mixed.mixed = True
    vec_env_mixed = make_vec_env(
        lambda: make_env(args_mixed),
        n_envs=args.num_processes,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    model_mixed = construct_MixedItemFarmPolicy(
        algorithm=args.algo,
        env=vec_env_mixed,
        device=fabric.device,
    )
    model_mixed.policy.policy_c = model.policy
    model_mixed.policy.policy_nc = model_normal.policy

    last_run_id = get_latest_run_id(log_path=args.log_dir, log_name=args.log_name)
    log_dir_actual = os.path.join(args.log_dir, f"{args.log_name}_{last_run_id + 1}")
    os.makedirs(log_dir_actual, exist_ok=True)

    model_mixed.save(os.path.join(log_dir_actual, f"rl_model_mixed.zip"))

    vec_env.close()
    vec_env_normal.close()


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Construct mixed agent")
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="training accelerator"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    add_argument_environment(parser)
    add_argument_algorithm(parser)
    add_argument_train_agent(parser)
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--checkpoint_normal",
        type=str,
        default="",
        help="checkpoint of the normal model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
