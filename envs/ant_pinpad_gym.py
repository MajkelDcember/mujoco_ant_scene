"""
AntPinpadGym — Gymnasium wrapper around AntPinpad.

Observation : raw ant obs (env.obs_dim) + 4D direction augmentation = obs_dim + 4
Action      : Box(low=-1, high=1, shape=(8,), dtype=float32)
Reward      : env_reward + intrinsic - posture_penalty + survival

Per-step diagnostics added to info dict (available to callbacks):
    speed       float   velocity magnitude (m/s)
    align       float   dot product of velocity with target direction
    idle        float   1.0 if speed < 0.05 else 0.0
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ant_pinpad import AntPinpad, get_current_direction, get_direction_augmentation
from tasks import ANT_PRETRAINING_TASKS, ANT_CONFIG


class AntPinpadGym(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        intrinsic_scale:           float = 12.0,
        posture_penalty_coef:      float = 10.0,
        posture_penalty_threshold: float = 0.7,
        subgoal_reward:            float = 2.0,
        survival_reward:           float = 0.1,
        fall_penalty:              float = 10.0,
        seed:                      Optional[int] = None,
    ):
        super().__init__()

        self._intrinsic_scale           = intrinsic_scale
        self._posture_penalty_coef      = posture_penalty_coef
        self._posture_penalty_threshold = posture_penalty_threshold
        self._subgoal_reward            = subgoal_reward
        self._survival_reward           = survival_reward
        self._fall_penalty              = fall_penalty
        self._seed                      = seed

        # Probe obs_dim from a throwaway env
        _kw  = {k: v for k, v in ANT_CONFIG.items() if k != "n_colors"}
        _tmp = AntPinpad(ANT_PRETRAINING_TASKS[0], **_kw, seed=0)
        raw_obs_dim = _tmp.obs_dim
        del _tmp

        obs_dim = raw_obs_dim + 4  # +4 for direction augmentation

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self._env_kwargs = {k: v for k, v in ANT_CONFIG.items() if k != "n_colors"}
        self._env_kwargs["subgoal_reward"]    = subgoal_reward
        self._env_kwargs["wrong_color_kills"] = False
        self._env: Optional[AntPinpad] = None
        self._rng = np.random.default_rng(seed)

        self._DT    = 0.05
        self._V_CAP = 1.0
        self._DIR_VECS = np.array(
            [[0., -1.], [0., 1.], [-1., 0.], [1., 0.]], dtype=np.float32
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _make_env(self) -> AntPinpad:
        task = ANT_PRETRAINING_TASKS[int(self._rng.integers(len(ANT_PRETRAINING_TASKS)))]
        return AntPinpad(task, **self._env_kwargs, seed=int(self._rng.integers(1_000_000)))

    def _aug_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        direction = get_direction_augmentation(self._env)
        return np.concatenate([raw_obs, direction], dtype=np.float32)

    def _shaped_reward(
        self,
        env_reward: float,
        direction:  np.ndarray,
        xy_prev:    tuple,
        xy_curr:    tuple,
        up_z:       float,
        fell:       bool = False,
    ) -> tuple[float, float, float, float]:
        """
        Returns (shaped_reward, raw_speed, align, capped_speed).
        Speed/align computed here so step() doesn't repeat the math.
        """
        vx = (xy_curr[0] - xy_prev[0]) / self._DT
        vy = (xy_curr[1] - xy_prev[1]) / self._DT
        speed = float(np.hypot(vx, vy))

        target = self._DIR_VECS.T @ direction
        align  = float(np.dot([vx, vy], target))   # raw (uncapped) align

        if speed > 1e-6:
            k  = min(1.0, self._V_CAP / speed)
            vx = vx * k
            vy = vy * k
        else:
            vx = vy = 0.0

        intrinsic       = self._intrinsic_scale * float(np.dot([vx, vy], target))
        posture_penalty = self._posture_penalty_coef * max(
            0.0, self._posture_penalty_threshold - up_z
        )
        survival         = 0.0 if fell else self._survival_reward
        terminal_penalty = self._fall_penalty if fell else 0.0
        shaped = env_reward + intrinsic - posture_penalty + survival - terminal_penalty

        return shaped, speed, align

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._env = self._make_env()
        raw_obs   = self._env.reset()
        return self._aug_obs(raw_obs), {}

    def step(self, action: np.ndarray):
        xy_prev   = self._env._get_torso_xy()
        direction = get_current_direction(self._env)

        raw_obs, env_reward, done, info = self._env.step(action)

        xy_curr = self._env._get_torso_xy()
        up_z    = float(self._env._torso_up_from_quat()[2])

        shaped_reward, speed, align = self._shaped_reward(
            env_reward, direction, xy_prev, xy_curr, up_z,
            fell=bool(info.get("fell", False))
        )

        # ── per-step diagnostics ───────────────────────────────────────────────
        info["speed"] = speed
        info["align"] = align
        info["idle"]  = float(speed < 0.05)
        # ──────────────────────────────────────────────────────────────────────

        truncated  = bool(info.get("timeout", False))
        terminated = done and not truncated

        obs = self._aug_obs(raw_obs)

        if done:
            info["is_success"] = info.get("success", False)

        return obs, shaped_reward, terminated, truncated, info

    def render(self, camera: str = "perspective", width: int = 480, height: int = 480):
        if self._env is None:
            return None
        return self._env.render(camera=camera, width=width, height=height)