"""Evaluation script"""

import argparse
import logging

from lightning.fabric import Fabric
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

from esp_simulator.policies import load_BlackjackPolicy
from esp_simulator.utils.parser import parse_str2intlist
from esp_simulator.utils.vec_env import EvaluateVecEnv

from utils.algorithms import add_argument_algorithm
from utils.environments import add_argument_environment, make_env


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    logging.basicConfig(level=logging.INFO)

    fabric = Fabric(accelerator=args.accelerator, devices="auto", strategy="auto")

    # Load environment
    vec_env = make_vec_env(
        lambda: make_env(args),
        n_envs=args.num_repeats,
        vec_env_cls=EvaluateVecEnv,
    )

    # Load model
    model = load_BlackjackPolicy(
        algorithm=args.algo,
        path=args.checkpoint,
        device=fabric.device,
    )

    steps_list = np.array([], dtype=int)
    rewards_list = np.array([])
    for seed_env in tqdm(args.seed_envs):
        if model is not None:
            model.set_random_seed(args.seed)
        vec_env.seed(seed_env)

        steps = np.zeros(args.num_repeats, dtype=int)
        total_rewards = np.zeros(args.num_repeats)

        states = None
        dones = np.zeros(args.num_repeats, dtype=bool)
        obss = vec_env.reset()

        while not all(dones):
            actions, states = model.predict(
                observation=obss,
                state=states,
            )
            obss, rewards, dones_tmp, infos = vec_env.step(actions)

            for rdx in range(args.num_repeats):
                if dones[rdx]:
                    continue

                total_rewards[rdx] += rewards[rdx]
                steps[rdx] += 1
                dones[rdx] = dones_tmp[rdx]

        steps_list = np.append(steps_list, steps)
        rewards_list = np.append(rewards_list, total_rewards)

    vec_env.close()

    step_mean = np.mean(steps_list)
    step_std = np.std(steps_list)
    reward_mean = np.mean(rewards_list)
    reward_std = np.std(rewards_list)

    logging.info(f"[Num Steps] Mean: {step_mean:.4f}, Std: {step_std:.4f}")
    logging.info(f"[Reward] Mean: {reward_mean:.4f}, Std: {reward_std:.4f}")


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Evaluation script")
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="evaluation accelerator"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--seed_envs",
        type=str,
        default="12000-13999",
        help="list of random seeds for environment",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="the number of repetitions",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    add_argument_algorithm(parser)
    add_argument_environment(parser)
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    opts.seed_envs = parse_str2intlist(opts.seed_envs)
    main(opts)
