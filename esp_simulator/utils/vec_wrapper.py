import os
from typing import Callable

import imageio
from gymnasium import logger
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


class VecVideoRecorderWrapper(VecVideoRecorder):
    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
    ):
        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:  # pragma: no cover # assume gym interface
            metadata = temp_env.metadata

        self.env.metadata = metadata
        assert self.env.render_mode == "rgb_array", (
            f"The render_mode must be 'rgb_array', not {self.env.render_mode}"
        )

        self.frames_per_sec = self.env.metadata.get("render_fps", 30)

        self.record_video_trigger = record_video_trigger
        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames: list[np.ndarray] = []

    def _stop_recording(self) -> None:
        """Stop current recording and saves the video."""
        assert self.recording, (
            "_stop_recording was called, but no recording was started"
        )

        if len(self.recorded_frames) == 0:  # pragma: no cover
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            imageio.mimsave(
                self.video_path, self.recorded_frames, fps=self.frames_per_sec
            )

        self.recorded_frames = []
        self.recording = False
