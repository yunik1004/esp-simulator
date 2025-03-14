from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import MaybeCallback
from torch import nn


class AdversarialLearner:
    """Adversarial training of agent and classifier"""

    def __init__(
        self,
        agent: PPO,
        classifier: nn.Module,
    ) -> None:
        self.agent = agent
        self.classifier = classifier

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> None:
        iteration = 0

        total_timesteps, callback = self.agent._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.agent.env is not None

        while self.agent.num_timesteps < total_timesteps:
            # # TODO: model.separate_learn() ???
            continue_training = self.agent.collect_rollouts(
                self.agent.env,
                callback,
                self.agent.rollout_buffer,
                n_rollout_steps=self.agent.n_steps,
            )

            if not continue_training:
                break

            iteration += 1
            self.agent._update_current_progress_remaining(
                self.agent.num_timesteps, total_timesteps
            )

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.agent.ep_info_buffer is not None
                self.agent._dump_logs(iteration)

            self.agent.train()

        callback.on_training_end()
