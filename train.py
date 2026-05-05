"""
train.py
========
Custom training entrypoint that extends scene-deeprl-agents with
DiagnosticsCallback and VideoCallback.

Usage:
    uv run -m train project=ant_pinpad sim_name=... env=ant_pinpad model=ppo_ant_sbx train=ant_pinpad
"""

import os
import hashlib
import re
from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from scene_deeprl_agents.callbacks import InputReconstructionCallback, TensorboardCallback
from scene_deeprl_agents.model import Trainer
from scene_deeprl_agents.env import make_parallel_envs

from envs.callbacks import DiagnosticsCallback, VideoCallback


CHECKPOINT_RE = re.compile(r"rl_model_(\d+)_steps\.zip$")


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def _vecnormalize_path_for(checkpoint_path: Path) -> Path | None:
    step = _checkpoint_step(checkpoint_path)
    if step < 0:
        return None
    vecnormalize_path = checkpoint_path.with_name(f"rl_model_vecnormalize_{step}_steps.pkl")
    return vecnormalize_path if vecnormalize_path.exists() else None


def _find_latest_checkpoint(log_path: str) -> tuple[Path, Path | None] | None:
    current_run_dir = Path(log_path).resolve()
    sim_dir = current_run_dir.parent
    if not sim_dir.exists():
        return None

    candidates = [
        path
        for path in sim_dir.glob("*/rl_model_*_steps.zip")
        if path.parent.resolve() != current_run_dir and _checkpoint_step(path) >= 0
    ]
    if not candidates:
        return None
    checkpoint_path = max(candidates, key=lambda path: (_checkpoint_step(path), path.stat().st_mtime))
    return checkpoint_path, _vecnormalize_path_for(checkpoint_path)


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig) -> None:
    if cfg.sim_name is None:
        raise ValueError("sim_name must be set")

    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    resume_checkpoint = None
    if cfg.train.get("resume", True) and "path" not in cfg.model:
        resume_paths = _find_latest_checkpoint(log_path)
        if resume_paths is not None:
            resume_checkpoint, resume_vecnormalize = resume_paths
            with open_dict(cfg):
                cfg.model.path = str(resume_checkpoint)
                if resume_vecnormalize is not None:
                    cfg.env.path = str(resume_vecnormalize)
            print(f"[train] resuming model from {resume_checkpoint}")
            if resume_vecnormalize is not None:
                print(f"[train] resuming VecNormalize from {resume_vecnormalize}")
            else:
                print("[train] no matching VecNormalize checkpoint found; using fresh normalization stats")

    # stable 8-char ID derived from sim_name so restarts after preemption resume the same run
    run_id = hashlib.md5(cfg.sim_name.encode()).hexdigest()[:8]

    run = wandb.init(
        project=cfg.project,
        name=cfg.sim_name,
        id=run_id,
        resume="allow",
        config=dict(cfg),
        sync_tensorboard=True,
        dir=log_path,
        save_code=True,
    )
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")

    envs    = make_parallel_envs(cfg.env, log_path, path=cfg.env.path if "path" in cfg.env else None)
    trainer = Trainer(envs, cfg.model, log_path)
    num_envs = int(cfg.env.get("num_env", 1))
    total_timesteps = int(cfg.train.total_timesteps)
    rollout_size = num_envs * int(cfg.model.kwargs.n_steps)

    print(
        "[train] resolved "
        f"total_timesteps={total_timesteps:,} "
        f"num_envs={num_envs} "
        f"n_steps={int(cfg.model.kwargs.n_steps):,} "
        f"rollout_size={rollout_size:,} "
        f"updates~={total_timesteps / rollout_size:.1f}"
    )

    # ── build extra callbacks ────────────────────────────────────────────────
    env_kwargs = dict(OmegaConf.to_container(cfg.env.config, resolve=True)) if "config" in cfg.env else {}

    callbacks = []

    if cfg.train.get("save_model_checkpoints", False):
        # SB3's CheckpointCallback counts vector-env calls. Divide by num_envs so
        # train.save_freq keeps its natural meaning: total environment timesteps.
        checkpoint_freq = max(int(cfg.train.save_freq) // num_envs, 1)
        print(
            "[train] model checkpoints enabled "
            f"every {int(cfg.train.save_freq):,} env steps "
            f"({checkpoint_freq:,} callback calls)"
        )
        callbacks.append(
            CheckpointCallback(
                save_path=log_path,
                save_freq=checkpoint_freq,
                save_vecnormalize=isinstance(envs, VecNormalize),
                verbose=2,
            )
        )
    else:
        print("[train] model checkpoints disabled; avoiding SBX checkpoint serialization crash")

    if "reconstruction" in cfg.train and cfg.train.reconstruction is not None:
        callbacks.append(
            InputReconstructionCallback(
                save_path=os.path.join(log_path, "reconstruction_mlp.pkl"),
                vec_normalize=envs,
                **cfg.train.reconstruction,
            )
        )

    if "info_keywords" in cfg.train:
        callbacks.append(TensorboardCallback(info_keywords=cfg.train.info_keywords))

    callbacks.extend(
        [
            DiagnosticsCallback(),
            VideoCallback(
                env_kwargs  = env_kwargs,
                save_dir    = os.path.join(log_path, "videos"),
                video_every = cfg.train.get("video_every",      512_000),
                video_fps   = cfg.train.get("video_fps",        15),
                frame_skip  = cfg.train.get("video_frame_skip", 5),
                camera      = cfg.train.get("video_camera",     "perspective"),
                width       = cfg.train.get("video_width",      480),
                height      = cfg.train.get("video_height",     480),
                seed        = cfg.env.seed,
            ),
        ]
    )

    learn_kwargs = dict(OmegaConf.to_container(cfg.train.kwargs, resolve=True))
    if resume_checkpoint is not None:
        resumed_steps = int(getattr(trainer.agent, "num_timesteps", 0))
        remaining_timesteps = max(total_timesteps - resumed_steps, 0)
        learn_kwargs["reset_num_timesteps"] = False
        print(
            "[train] resume progress "
            f"checkpoint_steps={resumed_steps:,} "
            f"remaining_timesteps={remaining_timesteps:,}"
        )
        if remaining_timesteps <= 0:
            print("[train] checkpoint already reached requested total_timesteps; skipping learn()")
        else:
            trainer.agent.learn(total_timesteps=remaining_timesteps, callback=callbacks, **learn_kwargs)
    else:
        trainer.agent.learn(total_timesteps=total_timesteps, callback=callbacks, **learn_kwargs)

    if cfg.train.get("save_final_model", False):
        try:
            trainer.agent.save(os.path.join(log_path, "final_model.pkl"))
        except TypeError as e:
            if "cannot pickle" in str(e):
                print("[warning] Final model save failed due to pickle error.")
            else:
                raise
    else:
        print("[train] final model save disabled; avoiding SBX final serialization crash")

    if isinstance(envs, VecNormalize):
        envs.save(os.path.join(log_path, "final_env.pkl"))


if __name__ == "__main__":
    main()
