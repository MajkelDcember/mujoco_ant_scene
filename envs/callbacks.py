"""
callbacks.py
============
Custom callbacks for ant-pinpad training.

DiagnosticsCallback
    Logs per-episode and per-step metrics to W&B at the end of every rollout.

    Per-episode (only counted when an episode actually finishes):
        success_rate, fell_rate, hit_wall_rate, wrong_color_rate,
        timeout_rate, mean_subgoals, ep_rew_mean, ep_len_mean

    Per-step (averaged over every step in the rollout, not just episode ends):
        mean_speed, mean_align, idle_frac

VideoCallback
    Periodically runs a greedy episode, renders to mp4, logs to W&B.
    Annotates frames with current target color and direction.
    Fixed for SBX: np.asarray(action) before env.step().
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

from tasks import ANT_PRETRAINING_TASKS


class DiagnosticsCallback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # per-episode accumulators (only when episode ends)
        self._ep_rews:        list = []
        self._ep_lens:        list = []
        self._ep_succ:        list = []
        self._ep_fell:        list = []
        self._ep_hit_wall:    list = []
        self._ep_wrong_color: list = []
        self._ep_timeout:     list = []
        self._ep_subgoals:    list = []
        # per-step accumulators (every step)
        self._step_speed:     list = []
        self._step_align:     list = []
        self._step_idle:      list = []

        self._wandb_error_reported = False

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # ── per-step diagnostics (always present) ─────────────────────────
            self._step_speed.append(float(info.get("speed", 0.0)))
            self._step_align.append(float(info.get("align", 0.0)))
            self._step_idle.append(float(info.get("idle",  0.0)))

            # ── per-episode diagnostics (only when episode ends) ───────────────
            ep = info.get("episode")   # added by Monitor wrapper
            if ep is not None:
                self._ep_rews.append(float(ep["r"]))
                self._ep_lens.append(float(ep["l"]))
                self._ep_succ.append(float(info.get("is_success", False)))
                self._ep_fell.append(float(info.get("fell",        False)))
                self._ep_hit_wall.append(float(info.get("hit_wall",    False)))
                self._ep_wrong_color.append(float(info.get("wrong_color", False)))
                self._ep_timeout.append(float(info.get("timeout",    False)))
                self._ep_subgoals.append(float(info.get("subgoal_index", 0)))

        return True

    def _on_rollout_end(self) -> None:
        def _m(lst): return float(np.mean(lst)) if lst else 0.0

        try:
            import wandb
            log_dict = {}

            # per-step metrics — always log (full rollout coverage)
            if self._step_speed:
                log_dict["mean_speed"] = _m(self._step_speed)
                log_dict["mean_align"] = _m(self._step_align)
                log_dict["idle_frac"]  = _m(self._step_idle)

            # per-episode metrics — only log when we have complete episodes
            if self._ep_rews:
                log_dict.update({
                    "ep_rew_mean":      _m(self._ep_rews),
                    "ep_len_mean":      _m(self._ep_lens),
                    "success_rate":     _m(self._ep_succ),
                    "fell_rate":        _m(self._ep_fell),
                    "hit_wall_rate":    _m(self._ep_hit_wall),
                    "wrong_color_rate": _m(self._ep_wrong_color),
                    "timeout_rate":     _m(self._ep_timeout),
                    "mean_subgoals":    _m(self._ep_subgoals),
                })

            # SB3/SBX internal training losses from logger
            sb3_vals = getattr(self.model.logger, "name_to_value", {})
            for key in (
                "train/policy_gradient_loss", "train/value_loss",
                "train/entropy_loss", "train/approx_kl",
                "train/clip_fraction", "train/explained_variance",
                "train/clip_range", "train/learning_rate", "train/std",
            ):
                val = sb3_vals.get(key)
                if val is not None:
                    log_dict[key.replace("train/", "")] = float(val)

            if log_dict:
                wandb.log(log_dict, step=self.num_timesteps)

        except Exception as e:
            if not self._wandb_error_reported:
                print(f"[wandb] diagnostics logging failed: {e}")
                self._wandb_error_reported = True

        # clear all accumulators
        self._ep_rews.clear();        self._ep_lens.clear()
        self._ep_succ.clear();        self._ep_fell.clear()
        self._ep_hit_wall.clear();    self._ep_wrong_color.clear()
        self._ep_timeout.clear();     self._ep_subgoals.clear()
        self._step_speed.clear();     self._step_align.clear()
        self._step_idle.clear()


class VideoCallback(BaseCallback):
    """
    Periodically roll out a greedy episode, render to mp4, log to W&B.
    Annotates frames with current target color and direction.
    SBX-safe: action is cast to numpy before env.step().
    """

    _GOAL_NAMES = {0: "RED", 1: "BLUE", 2: "GREEN", 3: "YELLOW", -1: "DONE"}
    _GOAL_RGB   = {
        0: (230,  76,  61),
        1: ( 51, 153, 219),
        2: ( 46, 204, 113),
        3: (241, 196,  15),
       -1: (255, 255, 255),
    }
    _DIR_WORDS = {0: "DOWN", 1: "UP", 2: "LEFT", 3: "RIGHT"}

    def __init__(
        self,
        env_kwargs:  dict,
        save_dir:    str,
        video_every: int   = 512_000,
        video_fps:   int   = 15,
        frame_skip:  int   = 5,
        camera:      str   = "perspective",
        width:       int   = 480,
        height:      int   = 480,
        seed:        int   = 0,
        verbose:     int   = 0,
    ):
        super().__init__(verbose)
        self._env_kwargs  = env_kwargs
        self._save_dir    = Path(save_dir)
        self._video_every = video_every
        self._video_fps   = video_fps
        self._frame_skip  = frame_skip
        self._camera      = camera
        self._width       = width
        self._height      = height
        self._seed        = seed
        self._next_video  = video_every

    def _annotate(self, frame: np.ndarray, env) -> np.ndarray:
        import cv2
        inner   = env._env
        goal_id = inner.current_goal if inner._goal_idx < len(inner.task) else -1
        rgb     = self._GOAL_RGB.get(goal_id, (255, 255, 255))
        txt     = f"Target: {self._GOAL_NAMES.get(goal_id, 'UNKNOWN')}"
        cv2.putText(frame, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb,       2, cv2.LINE_AA)

        from ant_pinpad import get_current_direction
        direction = get_current_direction(inner)
        dir_idx   = int(np.argmax(direction)) if direction.sum() > 0 else -1
        dtxt      = f"Dir: {self._DIR_WORDS.get(dir_idx, 'NONE')}"
        cv2.putText(frame, dtxt, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),       4, cv2.LINE_AA)
        cv2.putText(frame, dtxt, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_video:
            return True

        try:
            import imageio
            import cv2  # noqa: F401
        except ImportError as e:
            print(f"[video] missing dependency, skipping: {e}")
            self._next_video += self._video_every
            return True

        from envs.ant_pinpad_gym import AntPinpadGym

        rng    = np.random.default_rng(self._seed + self.num_timesteps)
        env    = AntPinpadGym(**self._env_kwargs, seed=int(rng.integers(1_000_000)))
        obs, _ = env.reset()

        def _get_frame():
            frame = env.render(camera=self._camera, width=self._width, height=self._height)
            if frame is not None:
                return self._annotate(np.array(frame, copy=True), env)
            return None

        frames  = []
        f0      = _get_frame()
        if f0 is not None:
            frames.append(f0)

        done    = False
        step    = 0
        success = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action     = np.asarray(action)   # SBX returns JAX array — cast to numpy
            obs, _, terminated, truncated, info = env.step(action)
            done    = terminated or truncated
            success = info.get("is_success", False)
            step   += 1
            if step % self._frame_skip == 0 or done:
                frame = _get_frame()
                if frame is not None:
                    frames.append(frame)

        print(f"  [video] steps={self.num_timesteps:,}  T={step}  success={success}")

        if len(frames) > 1:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            mp4 = self._save_dir / f"traj_{self.num_timesteps:08d}.mp4"
            imageio.mimwrite(str(mp4), frames, fps=self._video_fps, quality=7)
            try:
                import wandb
                wandb.log({
                    "trajectory":      wandb.Video(str(mp4), fps=self._video_fps, format="mp4"),
                    "video_success":   float(success),
                    "video_ep_length": step,
                }, step=self.num_timesteps)
            except Exception:
                pass

        self._next_video += self._video_every
        return True
