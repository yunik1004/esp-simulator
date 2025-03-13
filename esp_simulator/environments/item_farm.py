from enum import Enum
from typing import Any, Dict, List, SupportsFloat, Tuple

from gymnasium import spaces
from gymnasium.core import ActType
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball
from minigrid.minigrid_env import MiniGridEnv
import numpy as np

from .actions import ItemFarmActions
from ..utils.tsp import parse_actions, solve_minigrid_tsp


class AgentType(Enum):
    PO = 1
    FO = 2


class BallItem(Ball):
    def can_overlap(self):
        return True


class BaseItemFarmEnv(MiniGridEnv):
    """Environment for simulating item farming game"""

    def __init__(
        self,
        grid_size=8,
        agent_start_pos: Tuple[int, int] | None = (1, 1),
        agent_start_dir: int | None = 0,
        max_steps: int | None = None,
        see_through_walls: bool = False,
        agent_view_size: int = 3,
        num_items: int = 5,
        **kwargs,
    ) -> None:
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_items = num_items
        self.num_items_remain = self.num_items

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None or max_steps < 0:
            max_steps = 4 * grid_size**2

        self.is_po = True
        highlight = True
        if agent_view_size < 0:
            agent_view_size = 3
            self.is_po = False
            highlight = False

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            highlight=highlight,
            **kwargs,
        )

        # Modify actions
        self.actions = ItemFarmActions
        self.action_space = spaces.Discrete(len(self.actions))

        # Modify observation space
        self.observation_space["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.grid.width, self.grid.height, 3),
            dtype="uint8",
        )
        self.observation_space["item"] = spaces.Discrete(self.num_items)

        # Items
        self.items = []

        # logs
        self.log_action = []
        self.log_agent_pos = []

    @staticmethod
    def _gen_mission():
        return "Pickup balls"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Save current start pos and dir
        self.agent_start_pos_cur = self.agent_pos
        self.agent_start_dir_cur = self.agent_dir

        # Place items
        self.items = []

        num_grids = (width - 2) * (height - 2)
        if num_grids < self.num_items + 1:
            raise ValueError(f"Too many items: {self.num_items}")

        lex_pos = np.arange(num_grids)
        lex_pos = np.delete(
            lex_pos, (self.agent_pos[0] - 1) + (self.agent_pos[1] - 1) * (width - 2)
        )

        items_lex_pos = (
            np.random.choice(lex_pos, self.num_items, replace=False)
            if self._np_random is None
            else self._np_random.choice(lex_pos, self.num_items, replace=False)
        )

        for lex_pos in items_lex_pos:
            w_pos = lex_pos % (width - 2) + 1
            h_pos = lex_pos // (width - 2) + 1
            self.put_obj(BallItem("green"), w_pos, h_pos)
            self.items.append((w_pos, h_pos))

        self.num_items_remain = self.num_items
        self.optimal_steps, self.optimal_path, self.optimal_actions = self.solve_tsp()

        # Reset logs
        self.log_action = []
        self.log_agent_pos = []

        if self.is_po:
            # Initialize highlight mask
            self.highlight_mask = np.zeros(
                (self.grid.width, self.grid.height), dtype=bool
            )
            self._update_highlight_mask()

    def gen_obs(self):
        obs = super().gen_obs()

        # Change image in observation
        image = self.grid.encode()

        # Add agent
        image[self.agent_pos] = (
            OBJECT_TO_IDX["agent"],
            COLOR_TO_IDX["red"],
            self.agent_dir,
        )

        obs["image"] = image
        if self.is_po:
            # Masking unseen regions
            obs["image"][~self.highlight_mask] = (OBJECT_TO_IDX["unseen"], 0, 0)

        obs["item"] = self.num_items - self.num_items_remain
        return obs

    def step(self, action):
        reward, terminated, truncated, info = self._step_main(action)

        if self.is_po:
            # Update highlight mask
            self._update_highlight_mask()

        # Save logs
        self.log_action.append(action)
        self.log_agent_pos.append(self.agent_pos)

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, info

    def _step_main(
        self, action: ActType
    ) -> Tuple[SupportsFloat, bool, bool, Dict[str, Any]]:
        """Main parts of the step. Rendering or observations are not included

        Parameters
        ----------
        action : ActType
            an action provided by the agent to update the environment state.

        Returns
        -------
        tuple[SupportsFloat, bool, bool, dict[str, Any]]
            reward : SupportsFloat
            terminated : bool
            truncated : bool
            info : dict

        Raises
        ------
        ValueError
            When the action is not defined
        """
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "ball":
                reward = self._reward()
                self.grid.set(fwd_pos[0], fwd_pos[1], None)
                self.num_items_remain -= 1
                if self.num_items_remain <= 0:
                    terminated = True

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        info = {}
        if terminated or truncated:
            info["l_ratio"] = self.step_count / self.optimal_steps

        return reward, terminated, truncated, info

    def solve_tsp(
        self,
    ) -> Tuple[int, List[Tuple[Tuple[int, int], int]], List[ItemFarmActions]]:
        """Solve TSP problem

        Returns
        -------
        Tuple[int, List[Tuple[Tuple[int, int], int]], List[ESPActions]]
            Optimal cost, Optimal path (List of (item pos, dir)), Optimal actions
        """
        optimal_cost, optimal_path = solve_minigrid_tsp(
            self.agent_start_pos_cur,
            self.agent_start_dir_cur,
            self.items,
            self.max_steps,
        )
        if optimal_cost > self.max_steps:
            optimal_cost = -1
        optimal_path_pos = [(self.items[idx], dir) for idx, dir in optimal_path]

        optimal_actions = []
        start_pos, start_dir = self.agent_start_pos_cur, self.agent_start_dir_cur
        for path in optimal_path_pos:
            end_pos, end_dir = path
            optimal_actions += parse_actions(start_pos, start_dir, end_pos, end_dir)
            start_pos, start_dir = end_pos, end_dir

        return optimal_cost, optimal_path_pos, optimal_actions

    def visualize_status(self) -> dict[str, np.ndarray]:
        """Visualize the path history and the items by map format

        Returns
        -------
        dict[str, np.ndarray]
            path, item, total
        """
        # Draw map of path: 1(start_pos), 2, 3, ...
        map_path = np.zeros((self.grid.width, self.grid.height), dtype=int)
        map_path[self.agent_start_pos_cur] = 1
        for step, pos in enumerate(self.log_agent_pos, 2):
            map_path[pos] = step

        # Draw map of items: 1
        map_item = np.zeros((self.grid.width, self.grid.height), dtype=bool)
        for pos_item in self.items:
            map_item[pos_item] = 1

        # Concatenate rescaled maps
        map_total = np.stack(
            (map_path.astype(float) / map_path.max(), map_item.astype(float))
        )

        maps = {
            "path": map_path,
            "item": map_item,
            "total": map_total,
        }
        return maps

    def _update_highlight_mask(self) -> None:
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                self.highlight_mask[abs_i, abs_j] = True

    def get_full_render(self, highlight, tile_size):
        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=self.highlight_mask if highlight else None,
        )
        return img
