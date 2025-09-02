from enum import IntEnum


class ItemFarmActions(IntEnum):
    """Actions for item farm environment"""

    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2


class BlackjackActions(IntEnum):
    """Actions for Blackjack environment"""

    hit = 0
    stand = 1
    double_down = 2
    surrender = 3
