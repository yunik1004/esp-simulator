"""Behavior cloning script"""

import argparse
import copy
import os

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger
from lightning.fabric import Fabric
from lightning.pytorch import seed_everything
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import SubprocVecEnv

from esp_simulator.policies import construct_ItemFarmPolicy
from esp_simulator.utils import bc

from utils.agents import add_argument_train_agent
from utils.algorithms import add_argument_algorithm
from utils.environments import add_argument_environment, make_env


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed, workers=True)

    fabric = Fabric(accelerator=args.accelerator, devices="auto", strategy="auto")

    vec_env = make_vec_env(
        lambda: make_env(args, post_wrappers=[lambda env: RolloutInfoWrapper(env)]),
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
        lambda: make_env(
            args_normal, post_wrappers=[lambda env: RolloutInfoWrapper(env)]
        ),
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

    last_run_id = get_latest_run_id(log_path=args.log_dir, log_name=args.log_name)
    log_dir_actual = os.path.join(args.log_dir, f"{args.log_name}_{last_run_id + 1}")
    bc_logger = logger.configure(log_dir_actual, ["stdout", "tensorboard"])

    bc_trainer = bc.BCWithValue(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        rng=np.random.default_rng(seed=args.seed),
        policy=model.policy,
        device=fabric.device,
        custom_logger=bc_logger,
    )

    batch_num, num_samples_so_far = 0, 0
    for epoch in range(args.num_bc_epochs):
        rollouts = bc.rollout_cheater(
            model_normal,
            vec_env_normal,
            rollout.make_sample_until(min_episodes=args.rollout_episodes),
            rng=np.random.default_rng(),
        )
        transitions = rollout.flatten_trajectories(rollouts)
        bc_trainer.set_demonstrations(transitions)

        batch_num, num_samples_so_far = bc_trainer.train(
            epoch_start=epoch,
            batch_num_start=batch_num,
            num_samples_so_far_start=num_samples_so_far,
            n_epochs=1,
        )
        num_step = bc_trainer._bc_logger._tensorboard_step - 1
        model.save(os.path.join(log_dir_actual, f"rl_model_{num_step}_steps.zip"))


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Train script")
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="training accelerator"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--num_bc_epochs",
        type=int,
        default=int(1e4),
        help="the number of bc epochs",
    )
    parser.add_argument(
        "--rollout_episodes",
        type=int,
        default=8192,
        help="the number of rollout episodes per epoch",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="checkpoint of the model",
    )
    parser.add_argument(
        "--checkpoint_normal",
        type=str,
        default="",
        help="checkpoint of the normal model",
    )

    add_argument_algorithm(parser)
    add_argument_environment(parser)
    add_argument_train_agent(parser)
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
