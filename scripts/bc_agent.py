"""Behavior cloning script"""

import argparse
import copy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from lightning.fabric import Fabric
from lightning.pytorch import seed_everything
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from esp_simulator.policies import construct_ItemFarmPolicy

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

    rollouts = rollout.rollout(
        model_normal,
        vec_env_normal,
        rollout.make_sample_until(min_episodes=50),
    )
    transitions = rollout.flatten_trajectories(rollouts)

    bc_trainer = bc.BC(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        policy=model,
        demonstrations=transitions,
    )

    bc_trainer.train(
        n_epochs=1,
    )


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
