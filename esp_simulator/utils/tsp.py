from typing import List, Tuple

import numpy as np

from ..environments.actions import ItemFarmActions


def solve_minigrid_tsp(
    agent_pos: Tuple[int, int],
    agent_dir: int,
    target_poses: List[Tuple[int, int]],
    max_steps: int,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Solve TSP in Minigrid setting

    Parameters
    ----------
    agent_pos : Tuple[int, int]
        Start position of the agent
    agent_dir : int
        Start direction of the agent
    target_poses : List[Tuple[int, int]]
        Positions of the targets
    max_steps : int
        Maximum number of steps

    Returns
    -------
    Tuple[int, List[Tuple[int, int]]]
        Optimal cost, Optimal path (List of (target index, dir))
    """
    num_targets = len(target_poses)

    dp = np.full((1 << num_targets, num_targets, 4), np.inf)
    parent = np.full((1 << num_targets, num_targets, 4, 2), -1, dtype=int)

    # Initialize
    optimal = {
        "visited": 0,
        "num_visited": 0,
        "cost": max_steps + 1,
        "last_node": -1,
        "last_dir": -1,
    }

    for tdx in range(num_targets):
        visited = 1 << tdx
        for dir in range(4):
            new_cost = calculate_cost_with_rotation(
                agent_pos, agent_dir, target_poses[tdx], dir
            )
            dp[visited][tdx][dir] = new_cost
            if new_cost <= max_steps and new_cost < optimal["cost"]:
                optimal = {
                    "visited": visited,
                    "num_visited": 1,
                    "cost": new_cost,
                    "last_node": tdx,
                    "last_dir": dir,
                }

    # Update DP table
    for visited in range(1 << num_targets):
        for u in range(num_targets):
            for dir_u in range(4):
                if not (visited & (1 << u)):  # If u is not visited, pass
                    continue

                for v in range(num_targets):
                    if visited & (1 << v):  # If v is already visited, pass
                        continue

                    new_visited = visited | (1 << v)
                    num_visited = count_one(new_visited)
                    for dir_v in range(4):
                        new_cost = dp[visited][u][dir_u] + calculate_cost_with_rotation(
                            target_poses[u], dir_u, target_poses[v], dir_v
                        )
                        if new_cost < dp[new_visited][v][dir_v]:
                            dp[new_visited][v][dir_v] = new_cost
                            parent[new_visited][v][dir_v] = [u, dir_u]

                            if new_cost <= max_steps:
                                if num_visited < optimal["num_visited"]:
                                    continue
                                elif (
                                    num_visited == optimal["num_visited"]
                                    and new_cost > optimal["cost"]
                                ):
                                    continue
                                optimal = {
                                    "visited": new_visited,
                                    "num_visited": num_visited,
                                    "cost": new_cost,
                                    "last_node": v,
                                    "last_dir": dir_v,
                                }

    # Find optimal path
    mask = optimal["visited"]
    optimal_cost = optimal["cost"]
    last_node = optimal["last_node"]
    last_dir = optimal["last_dir"]
    """
    mask = (1 << num_targets) - 1
    optimal_cost, last_node, last_dir = min(
        (dp[mask][tdx][dir], tdx, dir) for tdx in range(num_targets) for dir in range(4)
    )
    """

    optimal_path = []
    while last_node != -1:
        optimal_path.append((last_node, last_dir))
        parent_node, parent_dir = parent[mask][last_node][last_dir]
        mask ^= 1 << last_node
        last_node, last_dir = parent_node, parent_dir
    optimal_path = optimal_path[::-1]

    return int(optimal_cost), optimal_path


def calculate_cost_with_rotation(
    agent_pos: Tuple[int, int],
    agent_dir: int,
    target_pos: Tuple[int, int],
    target_dir: int,
) -> int:
    """Compute the number of actions to convert (agent_pos, agent_dir) to (target_pos, target_dir)

    Parameters
    ----------
    agent_pos : Tuple[int, int]
        Position of the agent
    agent_dir : int
        Direction of the agent
    target_pos : Tuple[int, int]
        Position of the target
    target_dir : int
        Direction of the target

    Returns
    -------
    int
        the number of actions
    """
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]

    # Target direction must be one of two directions
    change_dir_x = get_direction((dx, 0))
    change_dir_y = get_direction((0, dy))
    if target_dir not in [change_dir_x, change_dir_y]:
        return np.inf

    # Manhattan distance
    distance = abs(dx) + abs(dy)

    # Compute rotation cost
    if dx == 0 and dy == 0:
        rotation_cost = get_rotation_cost(agent_dir, target_dir)
    elif dx == 0 or dy == 0:
        rotation_cost = get_rotation_cost(agent_dir, target_dir)
    else:
        rotation_cost_1 = (
            get_rotation_cost(agent_dir, change_dir_x)
            + get_rotation_cost(change_dir_y, target_dir)
            + 1
        )
        rotation_cost_2 = (
            get_rotation_cost(agent_dir, change_dir_y)
            + get_rotation_cost(change_dir_x, target_dir)
            + 1
        )
        rotation_cost = min(rotation_cost_1, rotation_cost_2)

    total_cost = distance + rotation_cost
    return total_cost


def get_rotation_cost(start_dir: int, end_dir: int) -> int:
    """Compute the number of rotations to convert start_dir to end_dir

    Parameters
    ----------
    start_dir : int
        Start direction
    end_dir : int
        End direction

    Returns
    -------
    int
        The number of rotations to convert start_dir to end_dir
    """
    dir_diff = abs(start_dir - end_dir)
    return min(dir_diff, 4 - dir_diff)


def get_direction(vec: Tuple[int, int]) -> int:
    """Get direction from the vector

    Parameters
    ----------
    vec : Tuple[int, int]
        Vector

    Returns
    -------
    int
        direction (0: east, 1: south, 2: west, 3: north, -1: no direction)
    """
    if vec[0] > 0:
        return 0  # East
    elif vec[0] < 0:
        return 2  # West
    elif vec[1] > 0:
        return 1  # South
    elif vec[1] < 0:
        return 3  # North
    return -1  # No direction


def parse_actions(
    agent_pos: Tuple[int, int],
    agent_dir: int,
    target_pos: Tuple[int, int],
    target_dir: int,
) -> List[ItemFarmActions]:
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]

    if dx == 0 or dy == 0:
        actions = parse_actions_dir(agent_dir, target_dir)
        actions += [ItemFarmActions.forward] * (abs(dx) + abs(dy))
        return actions
    else:
        change_dir_1 = get_direction((dx, 0))
        d1 = abs(dx)
        change_dir_2 = get_direction((0, dy))
        d2 = abs(dy)
        if change_dir_1 == target_dir:
            change_dir_1, change_dir_2 = change_dir_2, change_dir_1
            d1, d2 = d2, d1

        actions = parse_actions_dir(agent_dir, change_dir_1)
        actions += [ItemFarmActions.forward] * d1
        actions += parse_actions_dir(change_dir_1, change_dir_2)
        actions += [ItemFarmActions.forward] * d2

        return actions


def parse_actions_dir(agent_dir: int, target_dir: int) -> List[ItemFarmActions]:
    cost = get_rotation_cost(agent_dir, target_dir)

    if cost == 0:
        return []
    elif cost == 2:
        return [ItemFarmActions.left, ItemFarmActions.left]
    else:  # cost == 1
        if (agent_dir + 1) % 4 == target_dir:
            return [ItemFarmActions.right]
        else:
            return [ItemFarmActions.left]


def count_one(x: int) -> int:
    cnt = 0
    while x:
        x &= x - 1
        cnt += 1
    return cnt
