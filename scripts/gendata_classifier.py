import argparse
import itertools

from lightning.fabric import Fabric
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

from esp_simulator.environments.actions import ItemFarmActions
from esp_simulator.policies import load_ItemFarmPolicy
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
    fabric = Fabric(accelerator=args.accelerator, devices="auto", strategy="auto")

    vec_env = make_vec_env(
        lambda: make_env(args),
        n_envs=args.num_repeats,
        vec_env_cls=EvaluateVecEnv,
    )

    model = None
    if not args.optimal:
        model = load_ItemFarmPolicy(
            algorithm=args.algo,
            path=args.checkpoint,
            device=fabric.device,
        )

    maps = []
    traj_lengths = []
    total_rewards = []
    for seed_env in tqdm(args.seed_envs):
        if not args.optimal:
            model.set_random_seed(args.seed_policy)
        vec_env.seed(seed_env)

        states = None
        dones = np.zeros(args.num_repeats, dtype=bool)
        trewards = np.zeros(args.num_repeats)
        obss = vec_env.reset()

        optimal_steps = vec_env.env_method("get_wrapper_attr", "optimal_steps")
        optimal_actions = vec_env.env_method("get_wrapper_attr", "optimal_actions")

        for step in itertools.count():
            if args.optimal:
                actions = [
                    (
                        optimal_actions[rdx][step]
                        if step < optimal_steps[rdx]
                        else ItemFarmActions.left
                    )
                    for rdx in range(args.num_repeats)
                ]
            else:
                actions, states = model.predict(
                    observation=obss,
                    state=states,
                )

            obss, rewards, dones_tmp, infos = vec_env.step(actions)
            trewards += rewards

            for rdx in range(args.num_repeats):
                if dones[rdx]:
                    continue
                elif dones_tmp[rdx]:
                    dones[rdx] = True
                    map_total = infos[rdx]["visual_status"]
                    maps.append(map_total)
                    traj_lengths.append(step + 1)
                    total_rewards.append(trewards[rdx])

            if all(dones):
                break

    vec_env.close()

    # Save maps, traj lengths, total rewards in npy file format
    maps = np.stack(maps)
    np.save(args.output_path, maps)

    traj_lengths = np.stack(traj_lengths)
    np.save(args.output_path.replace(".npy", "_traj_len.npy"), traj_lengths)

    total_rewards = np.stack(total_rewards)
    np.save(args.output_path.replace(".npy", "_total_rew.npy"), total_rewards)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Script for generating data")
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="inference accelerator"
    )
    parser.add_argument("--optimal", action="store_true", help="optimal solution mode")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output/output.npy",
        help="output data path",
    )
    parser.add_argument(
        "--seed_envs",
        type=str,
        default="0-9999",
        help="list of random seeds for environment",
    )
    parser.add_argument(
        "--seed_policy", type=int, default=0, nargs="?", help="random seed for policy"
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="the number of repetitions",
    )
    add_argument_algorithm(parser)
    add_argument_environment(parser)
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    opts.seed_envs = parse_str2intlist(opts.seed_envs)
    main(opts)
