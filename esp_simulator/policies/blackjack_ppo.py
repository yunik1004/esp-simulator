from stable_baselines3 import PPO

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
    ItemFarmPPO
        ItemFarmPPO object
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
