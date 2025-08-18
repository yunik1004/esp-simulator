from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
            nn.Conv1d(3, 1, 1),
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
