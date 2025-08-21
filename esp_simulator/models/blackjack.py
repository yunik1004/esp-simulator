from typing import Dict

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
import torch
from torch import nn


class BlackjackFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=8,
    ):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 3, 1),
            nn.ReLU(),
            nn.Conv1d(3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.cnn(observations))


class BlackjackMultiFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Blackjack environment, multiple-input case
    Refer to: stable_baselines3.common.torch_layers.CombinedExtractor
    """

    def __init__(
        self,
        observation_space,
        features_dim=8,
        normalized_image=False,
        image_features_extractor: BaseFeaturesExtractor = BlackjackFeaturesExtractor,
    ):
        super().__init__(observation_space, 1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if "image" in key:
                extractors[key] = image_features_extractor(
                    subspace,
                    features_dim=features_dim,
                )
                total_concat_size += features_dim
            else:
                # The observation key is a vector, flatten it if needed
                obs_dim = get_flattened_obs_dim(subspace)
                extractors[key] = nn.Sequential(
                    nn.Flatten(), nn.Linear(obs_dim, obs_dim)
                )
                total_concat_size += obs_dim

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
