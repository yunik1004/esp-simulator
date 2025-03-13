"""Train script"""

import argparse
import os

from lightning.fabric import Fabric
from lightning.pytorch import seed_everything
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from esp_simulator.policies.item_farm import construct_ItemFarmPPO

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

    model = construct_ItemFarmPPO(
        env=vec_env,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        n_epochs=args.num_epochs,
        tensorboard_log=args.log_dir,
        verbose=1,
        device=fabric.device,
    )

    last_run_id = get_latest_run_id(log_path=args.log_dir, log_name=args.log_name)
    log_dir_actual = os.path.join(args.log_dir, f"{args.log_name}_{last_run_id + 1}")

    model.learn(
        total_timesteps=args.num_timesteps,
        callback=CallbackList(
            [
                CheckpointCallback(
                    save_freq=args.checkpoint_freq,
                    save_path=log_dir_actual,
                ),
            ]
        ),
        tb_log_name=args.log_name,
    )

    vec_env.close()


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
    parser.add_argument(
        "--log_dir", type=str, default="./output", help="tensorboard log dir"
    )
    parser.add_argument(
        "--log_name", type=str, default="ESPSimul", help="tensorboard log name"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="entropy coefficient"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=4, help="the number of epochs"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=16,
        help="the number of concurrent environments",
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=int(3e7), help="the number of timesteps"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=int(1e4),
        help="checkpoint saving frequency",
    )
    add_argument_environment(parser)
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
