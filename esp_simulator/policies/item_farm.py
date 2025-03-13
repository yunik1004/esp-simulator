from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.utils import safe_mean

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


def construct_ItemFarmPPO(features_dim: int = 64, *args, **kwargs) -> ItemFarmPPO:
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

    model = ItemFarmPPO(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        *args,
        **kwargs,
    )

    return model
