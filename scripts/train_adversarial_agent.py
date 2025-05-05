"""Train script"""

import argparse
import copy
import os

from lightning.fabric import Fabric
from lightning.pytorch import seed_everything
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from torch import nn
from torch.optim import Adam

from esp_simulator.models.item_farm import ItemFarmAntiCheater
from esp_simulator.policies import construct_ItemFarmPolicy

from utils.agents import add_argument_train_agent
from utils.algorithms import add_argument_algorithm
from utils.classifiers import add_argument_classifier
from utils.environments import add_argument_environment, make_env


class ClassifierCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        classifier: nn.Module,
        name_prefix: str = "classifier",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.classifier = classifier
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        return os.path.join(
            self.save_path,
            f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}",
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="pt")
            torch.save(self.classifier.state_dict(), model_path)
        return True


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
        learning_rate=args.lr,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        n_epochs=args.num_epochs,
        gamma=args.gamma,
        tensorboard_log=args.log_dir,
        verbose=1,
        device=fabric.device,
        adversary=True,
        adv_coef=args.adv_coef,
        cls_freq=args.cls_freq,
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

    classifier = ItemFarmAntiCheater(
        grid_size=args.grid_size, features_dim=args.features_dim
    )
    if args.checkpoint_classifier != "":
        checkpoint = torch.load(args.checkpoint_classifier)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        classifier.load_state_dict(checkpoint)
    classifier.to(fabric.device)

    optimizer_classifier = (
        Adam(classifier.parameters(), lr=args.lr_classifier)
        if args.train_classifier
        else None
    )

    last_run_id = get_latest_run_id(log_path=args.log_dir, log_name=args.log_name)
    log_dir_actual = os.path.join(args.log_dir, f"{args.log_name}_{last_run_id + 1}")

    callbacks = [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=log_dir_actual,
        ),
    ]
    if args.train_classifier:
        callbacks.append(
            ClassifierCheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=log_dir_actual,
                classifier=classifier,
            )
        )

    model.adversarial_learn(
        total_timesteps=args.adv_num_timesteps,
        callback=CallbackList(callbacks),
        tb_log_name=args.log_name,
        classifier=classifier,
        agent_normal=model_normal,
        optimizer_classifier=optimizer_classifier,
        method=args.method,
    )

    vec_env.close()
    vec_env_normal.close()


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
    add_argument_environment(parser)
    add_argument_algorithm(parser)
    add_argument_train_agent(parser)
    add_argument_classifier(parser)
    parser.add_argument(
        "--adv_num_timesteps",
        type=int,
        default=int(4e8),
        help="the number of timesteps in adversarial training mode",
    )
    parser.add_argument(
        "--adv_coef", type=float, default=0.1, help="adversarial training coefficient"
    )
    parser.add_argument(
        "--cls_freq", type=int, default=1, help="classifier update frequency"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--checkpoint_normal",
        type=str,
        default="",
        help="checkpoint of the normal model",
    )
    parser.add_argument(
        "--checkpoint_classifier",
        type=str,
        default="",
        help="checkpoint of the classifier",
    )
    parser.add_argument(
        "--lr_classifier",
        type=float,
        default=3e-4,
        help="learning rate of the classifier",
    )
    parser.add_argument(
        "--train_classifier",
        action="store_true",
        help="train the classifier",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="reward_shaping",
        choices=["reward_shaping", "policy_gradient"],
        help="adversarial training method",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
