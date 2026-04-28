"""
train.py
========
Custom training entrypoint that extends scene-deeprl-agents with
DiagnosticsCallback and VideoCallback.

Usage:
    uv run -m train project=ant_pinpad sim_name=... env=ant_pinpad model=ppo_ant_sbx train=ant_pinpad
"""

import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from scene_deeprl_agents.model import Trainer
from scene_deeprl_agents.env import make_parallel_envs

from envs.callbacks import DiagnosticsCallback, VideoCallback


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig) -> None:
    if cfg.sim_name is None:
        raise ValueError("sim_name must be set")

    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    wandb.init(
        project=cfg.project,
        name=cfg.sim_name,
        config=dict(cfg),
        sync_tensorboard=True,
        dir=log_path,
        save_code=True,
    )

    envs    = make_parallel_envs(cfg.env, log_path, path=cfg.env.path if "path" in cfg.env else None)
    trainer = Trainer(envs, cfg.model, log_path)

    # ── build extra callbacks ────────────────────────────────────────────────
    env_kwargs = dict(OmegaConf.to_container(cfg.env.config, resolve=True)) if "config" in cfg.env else {}

    diag_cb = DiagnosticsCallback()

    video_cb = VideoCallback(
        env_kwargs  = env_kwargs,
        save_dir    = os.path.join(log_path, "videos"),
        video_every = cfg.train.get("video_every",      512_000),
        video_fps   = cfg.train.get("video_fps",        15),
        frame_skip  = cfg.train.get("video_frame_skip", 5),
        camera      = cfg.train.get("video_camera",     "perspective"),
        width       = cfg.train.get("video_width",      480),
        height      = cfg.train.get("video_height",     480),
        seed        = cfg.env.seed,
    )

    # ── inject into trainer.agent.learn via monkey-patch ────────────────────
    # Trainer.train() calls self.agent.learn(callback=[checkpoint, wandb, ...])
    # We wrap learn() to append our callbacks to whatever list it already builds.
    original_learn = trainer.agent.learn

    def learn_with_extra(*args, **kwargs):
        existing = kwargs.get("callback", [])
        if not isinstance(existing, list):
            existing = [existing]
        kwargs["callback"] = existing + [diag_cb, video_cb]
        return original_learn(*args, **kwargs)

    trainer.agent.learn = learn_with_extra

    try:
        trainer.train(cfg.train)
    except TypeError as e:
        if "cannot pickle" in str(e):
            print("[warning] Final model save failed due to pickle error — checkpoints are still saved.")
        else:
            raise


if __name__ == "__main__":
    main()
