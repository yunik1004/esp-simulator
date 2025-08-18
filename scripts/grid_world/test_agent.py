"""Inference script"""

import argparse
import itertools
import logging

from lightning.fabric import Fabric
from minigrid.manual_control import ManualControl
import numpy as np
from PIL import Image

from esp_simulator.environments.actions import ItemFarmActions
from esp_simulator.policies import load_ItemFarmPolicy
from esp_simulator.utils.vec_env import InferenceVecEnv
from esp_simulator.utils.vec_wrapper import VecVideoRecorderWrapper

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

    # Manually controllable environment
    if args.checkpoint == "" and not args.optimal:
        args.interactive = True
        args.cheater = True
        env = make_env(args)
        env.reset(seed=args.seed_env)

        manual_control = ManualControl(env)
        manual_control.start()

        env.close()
        return

    if args.optimal:
        args.cheater = True

    # Load environment
    env = make_env(args)
    vec_env = InferenceVecEnv([lambda: env])

    if env.render_mode == "rgb_array":
        vec_env = VecVideoRecorderWrapper(
            vec_env,
            video_folder=args.output_dir,
            record_video_trigger=lambda _: True,
            video_length=env.unwrapped.max_steps,
        )

    vec_env.seed(args.seed_env)

    # Load model
    model = None
    if not args.optimal:
        model = load_ItemFarmPolicy(
            algorithm=args.algo,
            path=args.checkpoint,
            device=fabric.device,
        )
        model.set_random_seed(args.seed_policy)

    # Initialize the environment
    obs = vec_env.reset()
    state = None

    # Compute optimal solution
    optimal_steps = vec_env.envs[0].unwrapped.optimal_steps
    if optimal_steps < 0:
        optimal_steps = args.max_steps

    optimal_path = vec_env.envs[0].unwrapped.optimal_path
    optimal_actions = vec_env.envs[0].unwrapped.optimal_actions
    # logging.info(
    #     f"Num items: {len(optimal_path)}, Optimal steps: {optimal_steps}, Optimal path: {optimal_path}"
    # )

    # Simulate
    total_reward = 0
    for step in itertools.count():
        if args.optimal:
            if len(optimal_actions) > 0 and step < optimal_steps:
                action = [optimal_actions[step]]
            else:
                action = [ItemFarmActions.left]
        else:
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

        if env.render_mode == "human":
            vec_env.render(env.render_mode)

        if done:
            break

    # Save path as image
    if args.output_path_image != "":
        maps = vec_env.envs[0].unwrapped.visualize_status()
        map_traj = maps["total"][0].transpose()
        img = Image.fromarray(np.uint8(map_traj * 255), "L").resize(
            (352, 352), Image.Resampling.NEAREST
        )

        img.save(args.output_path_image)

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
    parser.add_argument("--optimal", action="store_true", help="optimal solution mode")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="output video dir"
    )
    parser.add_argument(
        "--output_path_image",
        type=str,
        default="",
        help="path of the output path image",
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
