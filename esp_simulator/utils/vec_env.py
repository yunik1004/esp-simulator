from copy import deepcopy
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class InferenceVecEnv(DummyVecEnv):
    """Disable reset from DummyVecEnv when calling step"""

    def step_wait(self):
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            (
                obs,
                self.buf_rews[env_idx],
                terminated,
                truncated,
                self.buf_infos[env_idx],
            ) = self.envs[env_idx].step(self.actions[env_idx])
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = (
                truncated and not terminated
            )

            # Disable reset
            """
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            """
            self._save_obs(env_idx, obs)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )


class EvaluateVecEnv(SubprocVecEnv):
    def seed(self, seed):
        """Sets the same random seed for all environments"""
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        self._seeds = [seed for _ in range(self.num_envs)]
        return self._seeds
