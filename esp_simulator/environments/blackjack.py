from itertools import product
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .actions import BlackjackActions


class BlackjackEnv(gym.Env):
    """Blackjack environment.
    The player has four options: hit, stand, double down, and surrender
    """

    def __init__(
        self,
        init_bet: float = 1.0,
        blackjack_ratio: float = 1.5,
        num_decks: int = 1,
        render_mode: str | None = None,
        is_cheater: bool = False,
        is_mixed: bool = False,
    ) -> None:
        self.init_bet = init_bet
        self.blackjack_ratio = blackjack_ratio
        self.num_decks = num_decks
        assert self.num_decks == 1

        self.render_mode = render_mode

        self.is_cheater = is_cheater
        self.is_mixed = is_mixed

        self.actions = BlackjackActions
        self.action_space = spaces.Discrete(len(self.actions))

        self.num_cards_required = 0
        if self.num_decks == 1:
            self.num_cards_required = (
                17  # (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,/ 3, 4, 4, 4, 4, 5)
            )
        else:
            NotImplementedError(f"Num_decks: {self.num_decks} is not supported")

        self.obs_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.is_ace = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # fmt: off
        self.trump_ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
        # fmt: on
        self.trump_suits = ["♤", "♡", "◇", "♧"]
        self.one_deck = [
            f"{r}{s}" for r, s in product(self.trump_ranks, self.trump_suits)
        ]
        self.one_deck_number = [
            n for n, _ in product(self.obs_number, self.trump_suits)
        ]
        self.one_deck_ace = [n for n, _ in product(self.is_ace, self.trump_suits)]
        if self.num_decks == 1:
            self.full_decks = [d for d in self.one_deck]
        else:
            self.full_decks = [
                f"{d}_{n}" for d, n in product(self.one_deck, range(self.num_decks))
            ]
        self.full_decks_number = [
            n for n, _ in product(self.one_deck_number, range(self.num_decks))
        ]
        self.full_decks_ace = [
            n for n, _ in product(self.one_deck_ace, range(self.num_decks))
        ]
        self.card_indices = list(range(len(self.full_decks_number)))

        # channel: obs_partial, check_ace_partial, state, check_ace_state, action
        # obs/state: dealer_1, (dealer_2), player_1, player_2, ..., (inv_1), (inv_2), ...
        image_observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(4, self.num_cards_required),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
            }
        )
        if self.is_mixed:
            self.observation_space["image_cheater"] = spaces.Box(
                low=0,
                high=10,
                shape=(4, self.num_cards_required),
                dtype="uint8",
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        self.step_count = 0
        self.step_base = 4
        self.total_bet = self.init_bet

        selected_indices = self.card_indices.copy()
        self.np_random.shuffle(selected_indices)
        self.selected_indices = selected_indices[: self.num_cards_required]

        self.selected_cards = [self.full_decks[i] for i in self.selected_indices]
        self.selected_numbers = [
            self.full_decks_number[i] for i in self.selected_indices
        ]
        self.selected_aces = [self.full_decks_ace[i] for i in self.selected_indices]

        self.state = np.array(
            [self.selected_numbers, self.selected_aces], dtype="uint8"
        )
        self.view_mask = np.array([False for _ in range(self.num_cards_required)])
        self.view_mask[0] = True
        self.view_mask[2] = True
        self.view_mask[3] = True

        self.prev_action = np.zeros((1, self.num_cards_required), dtype="uint8")
        obs = self.get_obs()

        return obs, {}

    def get_obs(self):
        partial_obs = np.where(self.view_mask, self.state, 0)

        obs = {}
        if self.is_cheater:
            obs["image"] = np.concatenate(
                [partial_obs[:1], self.state, self.prev_action], axis=0
            )
        else:
            obs["image"] = np.concatenate(
                [partial_obs[:1], partial_obs, self.prev_action], axis=0
            )

        if self.is_mixed:
            obs["image_cheater"] = np.concatenate(
                [partial_obs[:1], partial_obs, self.prev_action], axis=0
            )
        return obs

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}
        if action == BlackjackActions.hit:
            player_hand = self.selected_numbers[
                2 : self.step_base + self.step_count + 1
            ]
            self.view_mask[self.step_base + self.step_count] = True
            if sum(player_hand) > 21:
                terminated = True
                reward = -self.total_bet
        elif action == BlackjackActions.surrender:
            terminated = True
            if self.step_count == 0:
                reward = -0.5 * self.total_bet
            else:
                reward = -2 * self.total_bet
        else:
            terminated = True
            dealer_hand = self.selected_numbers[:2]
            player_hand = []
            remainder = []
            is_blackjack = False
            if action == BlackjackActions.stand:
                player_hand = self.selected_numbers[
                    2 : self.step_base + self.step_count
                ]
                remainder = self.selected_numbers[self.step_base + self.step_count :]
                if self.step_count == 0 and (
                    player_hand == [1, 10] or player_hand == [10, 1]
                ):  # blackjack
                    is_blackjack = True
            elif action == BlackjackActions.double_down:
                player_hand = self.selected_numbers[
                    2 : self.step_base + self.step_count + 1
                ]
                remainder = self.selected_numbers[
                    self.step_base + self.step_count + 1 :
                ]
                self.total_bet = self.total_bet * 2
                self.view_mask[self.step_base + self.step_count] = True

            player_min_score = sum(player_hand)
            player_max_score = player_min_score
            if player_min_score <= 11 and 1 in player_hand:
                player_max_score = player_min_score + 10

            if player_max_score > 21:
                reward = -self.total_bet
            else:
                # dealer phase
                dealer_min_score = sum(dealer_hand)
                dealer_max_score = dealer_min_score
                if dealer_min_score <= 11 and 1 in dealer_hand:
                    dealer_max_score = dealer_min_score + 10
                while dealer_max_score < 17:
                    dealer_hand.append(remainder.pop(0))
                    dealer_min_score = sum(dealer_hand)
                    dealer_max_score = dealer_min_score
                    if dealer_min_score <= 11 and 1 in dealer_hand:
                        dealer_max_score = dealer_min_score + 10

                if dealer_max_score > 21 or player_max_score > dealer_max_score:
                    if is_blackjack:
                        reward = self.total_bet * self.blackjack_ratio
                    else:
                        reward = self.total_bet
                elif player_max_score == dealer_max_score:
                    reward = 0
                else:
                    reward -= self.total_bet

        self.prev_action[0, self.step_base + self.step_count - 1] = action + 1

        obs = self.get_obs()

        self.step_count += 1

        return obs, reward, terminated, truncated, info

    def render(self) -> str:
        po_decks = [
            f"({card})" if not mask else card
            for card, mask in zip(self.selected_cards, self.view_mask)
        ]
        dealer_init_hand = po_decks[:2]
        player_init_hand = po_decks[2 : self.step_base]
        deck = po_decks[self.step_base :]
        actions = self.prev_action[0, self.step_base - 1 :]

        action_names = []
        for ac in actions:
            if ac == 0:
                continue
            action_names.append(BlackjackActions(ac - 1).name)

        output = f"Dealer: {dealer_init_hand}\nPlayer: {player_init_hand}\nDeck: {deck}\nActions: {action_names}"
        print(output)
        return output
