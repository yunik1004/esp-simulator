from collections import deque
from collections.abc import Generator
from typing import List

from gymnasium import spaces
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
import torch as th
from torch import nn
from torchmetrics import AUROC, AveragePrecision

from ..models.item_farm import ItemFarmMultiFeaturesExtractor


class ItemFarmA2C(A2C):
    """A2C policy for item farm environment"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ep_custom_buffer = None

    def _setup_learn(
        self,
        total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="run",
        ç=False,
    ):
        if self.ep_custom_buffer is None or reset_num_timesteps:
            self.ep_custom_buffer = deque(maxlen=self._stats_window_size)

        return super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, ç
        )

    def _update_info_buffer(self, infos, dones):
        assert self.ep_custom_buffer is not None

        for idx, info in enumerate(infos):
            maybe_l_ratio = info.get("l_ratio")
            if maybe_l_ratio is not None:
                self.ep_custom_buffer.extend([maybe_l_ratio])

        super()._update_info_buffer(infos, dones)

    def _dump_logs(self, iteration):
        assert self.ep_custom_buffer is not None

        if len(self.ep_custom_buffer) > 0:
            self.logger.record(
                "rollout/ep_len_ratio_mean", safe_mean(self.ep_custom_buffer)
            )

        super()._dump_logs(iteration)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def collect_visual_status(
        self,
        env: VecEnv,
        batch_size: int,
    ) -> List[th.Tensor]:
        """
        Collect visual status

        :param env: The training environment
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        output_buffer = []

        n_steps = 0
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        while len(output_buffer) < batch_size:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if done:
                    map = infos[idx]["visual_status"]
                    map = th.tensor(map, device=self.device).float()
                    output_buffer.append(map)

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        output_tensor = th.stack(output_buffer[:batch_size], dim=0)

        return output_tensor


def rollout_buffer_get_no_random(
    self: RolloutBuffer,
) -> Generator[RolloutBufferSamples, None, None]:
    assert self.full, ""
    indices = np.arange(self.buffer_size * self.n_envs)
    # Prepare the data
    if not self.generator_ready:
        if isinstance(self.observations, dict):
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)
        else:
            self.observations = self.swap_and_flatten(self.observations)

        _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

        for tensor in _tensor_names:
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
        self.generator_ready = True

    yield self._get_samples(indices)


class ItemFarmAdversarialA2C(ItemFarmA2C):
    def __init__(self, adv_coef: float = 0.0, adv_thre: float = 0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_coef = adv_coef
        self.adv_thre = adv_thre
        self.ep_adv_buffer = None
        self.ep_cls_buffer = None
        self.ep_auroc_buffer = None
        self.ep_ap_buffer = None

    def _setup_learn(
        self,
        total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="run",
        ç=False,
    ):
        if self.ep_adv_buffer is None or reset_num_timesteps:
            self.ep_adv_buffer = deque(maxlen=self._stats_window_size)
        if self.ep_cls_buffer is None or reset_num_timesteps:
            self.ep_cls_buffer = deque(maxlen=self._stats_window_size)
        if self.ep_auroc_buffer is None or reset_num_timesteps:
            self.ep_auroc_buffer = deque(maxlen=self._stats_window_size)
        if self.ep_ap_buffer is None or reset_num_timesteps:
            self.ep_ap_buffer = deque(maxlen=self._stats_window_size)

        return super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, ç
        )

    def _dump_logs(self, iteration):
        assert self.ep_adv_buffer is not None
        assert self.ep_cls_buffer is not None
        assert self.ep_auroc_buffer is not None
        assert self.ep_ap_buffer is not None

        if len(self.ep_adv_buffer) > 0:
            self.logger.record(
                "adversary/pred_cheater_mean", safe_mean(self.ep_adv_buffer)
            )
        if len(self.ep_cls_buffer) > 0:
            self.logger.record(
                "adversary/loss_classifier_mean", safe_mean(self.ep_cls_buffer)
            )
        if len(self.ep_auroc_buffer) > 0:
            self.logger.record(
                "adversary/auroc_classifier_mean", safe_mean(self.ep_auroc_buffer)
            )
        if len(self.ep_ap_buffer) > 0:
            self.logger.record(
                "adversary/ap_classifier_mean", safe_mean(self.ep_ap_buffer)
            )

        super()._dump_logs(iteration)

    def collect_adversarial_rollouts_reward_shaping(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        classifier: nn.Module,
        agent_normal: A2C,
        train_classifier: bool,
    ) -> tuple[bool, tuple[th.Tensor, th.Tensor]]:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        pred_buffer = []

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False, (None, None)

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

                if done:
                    # Adversarial reward
                    map = infos[idx]["visual_status"]
                    map = th.tensor(map, device=self.device).float().unsqueeze(0)
                    if train_classifier:
                        logit = classifier(map)
                    else:
                        with th.no_grad():
                            logit = classifier(map)
                    pred = th.sigmoid(logit)
                    pred_buffer.append(logit)

                    adv_penalty = logit.item() + np.log(
                        1 - np.min((pred.item(), self.adv_thre))
                    )
                    rewards[idx] -= self.adv_coef * adv_penalty

                    self.ep_adv_buffer.extend([pred.item()])

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        # Normal input prediction
        pred_cheater = th.concat(pred_buffer, dim=0)

        batch_size = pred_cheater.shape[0]
        vis_normal = agent_normal.collect_visual_status(
            env=agent_normal.env,
            batch_size=batch_size,
        )
        pred_normal = classifier(vis_normal)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True, (pred_cheater, pred_normal)

    def adversarial_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        classifier: nn.Module | None = None,
        agent_normal: A2C | None = None,
        optimizer_classifier: th.optim.Optimizer | None = None,
        method: str = "reward_shaping",
    ):
        assert classifier is not None
        assert agent_normal is not None
        assert method in ["reward_shaping"]
        train_classifier = optimizer_classifier is not None
        if train_classifier:
            classifier.train()
            criterion = nn.BCEWithLogitsLoss()
        else:
            classifier.eval()

        agent_normal._setup_learn(
            total_timesteps,
            None,
            reset_num_timesteps,
            f"{tb_log_name}_normal",
            False,
        )

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            if train_classifier:
                optimizer_classifier.zero_grad()

            continue_training, (pred_cheater, pred_normal) = (
                self.collect_adversarial_rollouts_reward_shaping(
                    self.env,
                    callback,
                    self.rollout_buffer,
                    n_rollout_steps=self.n_steps,
                    classifier=classifier,
                    agent_normal=agent_normal,
                    train_classifier=train_classifier,
                )
            )

            if not continue_training:
                break

            assert pred_cheater is not None
            assert pred_normal is not None

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

            # Compute AUROC and AP
            pred = th.concat([pred_cheater, pred_normal], dim=0)
            target = th.concat(
                [
                    th.ones_like(pred_cheater, dtype=th.long),
                    th.zeros_like(pred_normal, dtype=th.long),
                ],
                dim=0,
            )

            auroc_func = AUROC(task="binary")
            auroc = auroc_func(pred, target)
            self.ep_auroc_buffer.extend([auroc.item()])

            ap_func = AveragePrecision(task="binary")
            ap = ap_func(pred, target)
            self.ep_ap_buffer.extend([ap.item()])

            if train_classifier:
                loss = criterion(pred, target.float())
                loss.backward()

                optimizer_classifier.step()

                # Logging loss
                self.ep_cls_buffer.extend([loss.item()])

        callback.on_training_end()

        return self


def construct_ItemFarmA2C(
    features_dim: int = 64, adversary: bool = False, *args, **kwargs
) -> ItemFarmA2C:
    """Construct ItemFarmPPO policy with multiple-input feature extractor

    Parameters
    ----------
    features_dim : int, optional
        Feature dimension, by default 64

    Returns
    -------
    ItemFarmPPO
        ItemFarmPPO object
    """
    policy_kwargs = dict(
        features_extractor_class=ItemFarmMultiFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
        ),
    )

    algo = ItemFarmA2C if not adversary else ItemFarmAdversarialA2C

    model = algo(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        *args,
        **kwargs,
    )

    return model
