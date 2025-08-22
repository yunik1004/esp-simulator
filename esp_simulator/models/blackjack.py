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


class BlackjackAntiCheater(nn.Module):
    def __init__(self, num_cards_required: int = 17, features_dim: int = 8) -> None:
        super().__init__()

        n_input_channels = 3
        input_size = (n_input_channels, num_cards_required)

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
                torch.as_tensor(torch.randn(input_size)).unsqueeze(0).float()
            ).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1),
        )

    def forward(self, map: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.cnn(map)).squeeze(1)


class BlackjackTrajectoryLengthAntiCheater(nn.Module):
    def __init__(self, num_cards_required: int = 17) -> None:
        super().__init__()

        self.log_k = nn.Parameter(torch.tensor(0.0))  # sharpness
        self.threshold = nn.Parameter(torch.tensor(1.2))  # decision boundary

    def forward(self, length: torch.Tensor) -> torch.Tensor:
        k = torch.exp(self.log_k)
        return torch.sigmoid(k * (self.threshold - length))


class BlackjackRewardAntiCheater(nn.Module):
    def __init__(self, num_cards_required: int = 17) -> None:
        super().__init__()

        self.log_k = nn.Parameter(torch.tensor(0.0))  # sharpness
        self.threshold = nn.Parameter(torch.tensor(0.35))  # decision boundary

    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        k = torch.exp(self.log_k)
        return torch.sigmoid(k * (reward - self.threshold))
