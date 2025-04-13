from collections import deque
from collections.abc import Generator
from typing import List, Optional

from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
import torch as th
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC, AveragePrecision

from ..models.item_farm import ItemFarmMultiFeaturesExtractor


class ItemFarmPPO(PPO):
    """PPO policy for item farm environment"""

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


def rollout_buffer_get_with_index(
    self: RolloutBuffer,
    batch_size: Optional[int] = None,
) -> Generator[RolloutBufferSamples, None, None]:
    assert self.full, ""
    indices = np.random.permutation(self.buffer_size * self.n_envs)
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

    # Return everything, don't create minibatches
    if batch_size is None:
        batch_size = self.buffer_size * self.n_envs

    start_idx = 0
    while start_idx < self.buffer_size * self.n_envs:
        sampled_indices = indices[start_idx : start_idx + batch_size]
        yield self._get_samples(sampled_indices), sampled_indices
        start_idx += batch_size


class ItemFarmAdversarialPPO(ItemFarmPPO):
    def __init__(
        self,
        adv_coef: float = 0.0,
        adv_thre: float = 0.99,
        cls_freq: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adv_coef = adv_coef
        self.adv_thre = adv_thre
        self.cls_freq = cls_freq
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
        agent_normal: PPO,
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

                    # adv_penalty = logit.item() + np.log(1 - np.min((pred.item(), self.adv_thre)))
                    adv_penalty = np.log(np.exp(logit.item()) + 1)
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

    def collect_adversarial_rollouts_policy_gradient(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        classifier: nn.Module,
        agent_normal: PPO,
        train_classifier: bool,
    ) -> tuple[
        bool, tuple[th.Tensor, th.Tensor], tuple[th.Tensor | None, th.Tensor | None]
    ]:
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
        pos_prev = th.zeros(rollout_buffer.n_envs, dtype=th.long)
        penalty_buffer = th.zeros(
            rollout_buffer.buffer_size, rollout_buffer.n_envs, dtype=th.float32
        )
        penalty_mask = th.zeros(
            rollout_buffer.buffer_size, rollout_buffer.n_envs, dtype=th.float32
        )
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
                return False, (None, None), (None, None)

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
                    self.ep_adv_buffer.extend([pred.item()])

                    # adv_penalty = logit.item() + np.log(1 - np.min((pred.item(), self.adv_thre)))
                    adv_penalty = np.log(np.exp(logit.item()) + 1)
                    penalty_buffer[pos_prev[idx] : rollout_buffer.pos, idx] = (
                        adv_penalty
                    )
                    penalty_mask[pos_prev[idx] : rollout_buffer.pos, idx] = True
                    pos_prev[idx] = rollout_buffer.pos

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

        return True, (pred_cheater, pred_normal), (penalty_buffer, penalty_mask)

    def train_policy_gradient(
        self, penalty_buffer: th.Tensor, penalty_mask: th.Tensor
    ) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        penalty_buffer_flatten = self.rollout_buffer.swap_and_flatten(
            penalty_buffer
        ).squeeze()
        penalty_mask_flatten = self.rollout_buffer.swap_and_flatten(
            penalty_mask
        ).squeeze()

        penalty = penalty_buffer_flatten
        penalty_mean = th.mean(penalty[penalty_mask_flatten > 0.5])
        adv_penalty_full = (penalty - penalty_mean) * penalty_mask_flatten

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data, data_indices in rollout_buffer_get_with_index(
                self.rollout_buffer, self.batch_size
            ):
                adv = adv_penalty_full[data_indices]

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages

                # Update advantage function
                advantages -= self.adv_coef * adv.to(advantages.device)

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def adversarial_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        classifier: nn.Module | None = None,
        agent_normal: PPO | None = None,
        optimizer_classifier: th.optim.Optimizer | None = None,
        method: str = "reward_shaping",
    ):
        assert classifier is not None
        assert agent_normal is not None
        assert method in ["policy_gradient", "reward_shaping"]
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

            if method == "policy_gradient":
                (
                    continue_training,
                    (pred_cheater, pred_normal),
                    (penalty_buffer, penalty_mask),
                ) = self.collect_adversarial_rollouts_policy_gradient(
                    self.env,
                    callback,
                    self.rollout_buffer,
                    n_rollout_steps=self.n_steps,
                    classifier=classifier,
                    agent_normal=agent_normal,
                    train_classifier=train_classifier,
                )
            elif method == "reward_shaping":
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

            if method == "policy_gradient":
                self.train_policy_gradient(penalty_buffer, penalty_mask)
            elif method == "reward_shaping":
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

            if train_classifier and iteration % self.cls_freq == 0:
                loss = criterion(pred, target.float())
                loss.backward()

                optimizer_classifier.step()

                # Logging loss
                self.ep_cls_buffer.extend([loss.item()])

        callback.on_training_end()

        return self


def construct_ItemFarmPPO(
    features_dim: int = 64, adversary: bool = False, *args, **kwargs
) -> ItemFarmPPO:
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

    algo = ItemFarmPPO if not adversary else ItemFarmAdversarialPPO

    model = algo(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        *args,
        **kwargs,
    )

    return model
