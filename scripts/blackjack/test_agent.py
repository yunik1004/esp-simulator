"""Inference script"""

import argparse
import itertools
import logging

from lightning.fabric import Fabric

from esp_simulator.policies import load_BlackjackPolicy
from esp_simulator.utils.vec_env import InferenceVecEnv

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
    env = make_env(args)
    vec_env = InferenceVecEnv([lambda: env])

    vec_env.seed(args.seed_env)

    # Load model
    model = load_BlackjackPolicy(
        algorithm=args.algo,
        path=args.checkpoint,
        device=fabric.device,
    )
    model.set_random_seed(args.seed_policy)

    # Initialize the environment
    obs = vec_env.reset()
    state = None

    # Simulate
    total_reward = 0
    for step in itertools.count():
        action, state = model.predict(
            observation=obs,
            state=state,
            deterministic=args.seed_policy is None,
        )

        obs, reward, done, info = vec_env.step(action)

        total_reward += reward
        logging.info(
            f"[Step {step + 1}] Reward: {reward[0]:.4f} / Total reward: {total_reward[0]:.4f}"
        )

        if done:
            vec_env.render(env.render_mode)
            break

    vec_env.close()


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Test script")
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="inference accelerator"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--seed_env", type=int, nargs="?", help="random seed for environment"
    )
    parser.add_argument(
        "--seed_policy", type=int, nargs="?", help="random seed for policy"
    )
    add_argument_algorithm(parser)
    add_argument_environment(parser)
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
