from stable_baselines3 import PPO
from stable_baselines3.common.policies import (
    MultiInputActorCriticPolicy,
)

from .item_farm_ppo import ItemFarmMixedPolicy
from ..models.blackjack import BlackjackMultiFeaturesExtractor


class BlackjackAdversarialPPO(PPO):
    pass


def construct_BlackjackPPO(
    features_dim: int = 8, adversary: bool = False, *args, **kwargs
) -> PPO:
    """Construct BlackjackPPO policy with multiple-input feature extractor

    Parameters
    ----------
    features_dim : int, optional
        Feature dimension, by default 8

    Returns
    -------
    PPO
        PPO object
    """
    policy_kwargs = dict(
        features_extractor_class=BlackjackMultiFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
        ),
    )

    algo = PPO if not adversary else BlackjackAdversarialPPO

    model = algo(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        *args,
        **kwargs,
    )

    return model


def construct_MixedBlackjackPPO(
    features_dim: int = 8, adversary: bool = False, *args, **kwargs
) -> PPO:
    """Construct BlackjackPPO policy with multiple-input feature extractor

    Parameters
    ----------
    features_dim : int, optional
        Feature dimension, by default 8

    Returns
    -------
    PPO
        PPO object
    """
    policy_kwargs_sub = dict(
        features_extractor_class=BlackjackMultiFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
        ),
    )

    policy_kwargs = dict(
        features_extractor_class=BlackjackMultiFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
        ),
        policy_c=MultiInputActorCriticPolicy,
        policy_c_kwargs=policy_kwargs_sub,
        policy_nc=MultiInputActorCriticPolicy,
        policy_nc_kwargs=policy_kwargs_sub,
    )

    algo = PPO if not adversary else BlackjackAdversarialPPO

    model = algo(
        policy=ItemFarmMixedPolicy,
        policy_kwargs=policy_kwargs,
        *args,
        **kwargs,
    )

    return model
