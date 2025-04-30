from typing import Dict

import torch
from torch import nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict


class ItemFarmImageFeaturesExtractor(BaseFeaturesExtractor):
    """
    Image feature extractor for item farm environment
    Refer to: stable_baselines3.common.torch_layers.NatureCNN
    """

    def __init__(
        self,
        observation_space,
        features_dim=64,
        normalized_image=False,
    ):
        super().__init__(observation_space, features_dim)

        assert is_image_space(
            observation_space, check_channels=False, normalized_image=normalized_image
        )

        n_input_channels = observation_space.shape[0]
        self.cnn = self._construct_cnn(n_input_channels)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

    def _construct_cnn(self, n_input_channels: int) -> nn.Module:
        """Define CNN module to encode map

        Parameters
        ----------
        n_input_channels : int
            The number of input channels

        Returns
        -------
        nn.Module
            CNN module
        """
        return nn.Sequential(
            nn.Conv2d(n_input_channels, 16, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.Flatten(),
        )


class ItemFarmMultiFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for item farm environment, multiple-input case
    Refer to: stable_baselines3.common.torch_layers.CombinedExtractor
    """

    def __init__(
        self,
        observation_space,
        features_dim=64,
        normalized_image=False,
        image_features_extractor: BaseFeaturesExtractor = ItemFarmImageFeaturesExtractor,
    ):
        super().__init__(observation_space, 1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = image_features_extractor(
                    subspace,
                    features_dim=features_dim,
                    normalized_image=normalized_image,
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


class ItemFarmAntiCheater(nn.Module):
    def __init__(
        self, grid_size: int, n_input_channels: int = 2, features_dim: int = 4
    ) -> None:
        super().__init__()

        input_size = (n_input_channels, grid_size, grid_size)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, features_dim // 2, 2),
            nn.ReLU(),
            nn.Conv2d(features_dim // 2, features_dim, 2),
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
            nn.Linear(features_dim, 1),
        )

    def forward(self, map: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.cnn(map)).squeeze(1)
