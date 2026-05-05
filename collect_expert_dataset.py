"""
Collect clean expert trajectories from a trained SBX/SB3 ant-pinpad policy.

Default target is the two-wall bs2048 sweep submitted by submit_sweep.py.

Example:
    MUJOCO_GL=egl uv run --no-sync python collect_expert_dataset.py \
        --episodes-per-task 25 \
        --output outputs/datasets/ant_pinpad_w2_bs2048_expert.h5
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from ant_pinpad import AntPinpad, get_current_direction, _bfs_grid
from tasks import ANT_CONFIG, ANT_PRETRAINING_TASKS


CHECKPOINT_RE = re.compile(r"rl_model_(\d+)_steps\.zip$")


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def _find_latest_checkpoint(root: Path, pattern: str | None, contains: str) -> Path:
    # Avoid relying on shell-preserved glob patterns inside Run:AI commands.
    pattern = pattern or "rl_model_*_steps.zip"
    candidates = [
        path for path in root.rglob("rl_model_*_steps.zip")
        if (
            path.is_file()
            and _checkpoint_step(path) >= 0
            and (not contains or contains in str(path))
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints under {root} matched pattern={pattern!r} contains={contains!r}"
        )
    return max(candidates, key=lambda path: (_checkpoint_step(path), path.stat().st_mtime))


def _vecnormalize_for(checkpoint_path: Path) -> Path | None:
    step = _checkpoint_step(checkpoint_path)
    if step < 0:
        return None
    path = checkpoint_path.with_name(f"rl_model_vecnormalize_{step}_steps.pkl")
    return path if path.exists() else None


def _make_aug_obs(env: AntPinpad, raw_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([raw_obs, get_current_direction(env)], dtype=np.float32)


def _normalise_obs(vecnormalize, obs: np.ndarray) -> np.ndarray:
    if vecnormalize is None:
        return obs
    return vecnormalize.normalize_obs(obs[None, ...])[0]


def _clean_path_possible(env: AntPinpad) -> bool:
    """Return whether the sampled layout permits all subgoals without wrong colors."""
    current = env._current_cell()
    for target_color in env.task:
        target_cell = env._object_cells[target_color]
        blocked = set(map(tuple, env._wall_cells))
        for color, cell in env._object_cells.items():
            if color != target_color:
                blocked.add(tuple(cell))

        if current != target_cell:
            delta = _bfs_grid(current, target_cell, env.G, list(blocked))
            if delta is None:
                return False
        current = target_cell
    return True


def _layout_arrays(env: AntPinpad) -> tuple[np.ndarray, np.ndarray]:
    object_cells = np.zeros((env.O, 2), dtype=np.int16)
    for color in range(env.O):
        object_cells[color] = env._object_cells[color]
    wall_cells = np.asarray(env._wall_cells, dtype=np.int16)
    return object_cells, wall_cells


def _make_env(task: tuple[int, ...], seed: int, args: argparse.Namespace) -> AntPinpad:
    env_kwargs = dict(ANT_CONFIG)
    env_kwargs.update(
        n_walls=args.n_walls,
        subgoal_reward=args.subgoal_reward,
        wrong_color_kills=False,
        fall_penalty=args.fall_penalty,
    )
    return AntPinpad(task, **env_kwargs, seed=seed)


def _load_vecnormalize(vecnormalize_path: Path | None, args: argparse.Namespace):
    if vecnormalize_path is None:
        return None

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs.ant_pinpad_gym import AntPinpadGym

    def _env_fn():
        return AntPinpadGym(
            intrinsic_scale=args.intrinsic_scale,
            subgoal_reward=args.subgoal_reward,
            survival_reward=args.survival_reward,
            posture_penalty_coef=args.posture_penalty_coef,
            posture_penalty_threshold=args.posture_penalty_threshold,
            fall_penalty=args.fall_penalty,
            n_walls=args.n_walls,
            seed=args.seed,
        )

    vec_env = DummyVecEnv([_env_fn])
    vecnormalize = VecNormalize.load(str(vecnormalize_path), vec_env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return vecnormalize


def collect_one_attempt(model, vecnormalize, task: tuple[int, ...], seed: int, args: argparse.Namespace) -> dict:
    env = _make_env(task, seed, args)
    raw_obs = env.reset()
    clean_possible = _clean_path_possible(env)
    object_cells, wall_cells = _layout_arrays(env)

    if args.require_clean_path_possible and not clean_possible:
        return dict(
            task=np.asarray(task, dtype=np.int16),
            seed=np.int64(seed),
            clean_path_possible=np.bool_(False),
            clean_success=np.bool_(False),
            env_success=np.bool_(False),
            wrong_color_any=np.bool_(False),
            goals_reached=np.int16(0),
            length=np.int32(0),
            object_cells=object_cells,
            wall_cells=wall_cells,
            start_cell=np.asarray(env._start_cell, dtype=np.int16),
            final_cell=np.asarray(env._current_cell(), dtype=np.int16),
        )

    raw_observations = []
    observations = []
    policy_observations = []
    actions = []
    rewards = []
    dones = []
    directions = []
    current_goals = []
    subgoal_indices = []
    current_cells = []
    torso_xy = []
    qpos = []
    qvel = []
    ctrl = []
    goal_reached = []
    wrong_color = []
    hit_wall = []
    fell = []
    timeout = []

    wrong_color_any = False
    goals_reached = 0
    done = False
    final_info = {}

    while not done:
        aug_obs = _make_aug_obs(env, raw_obs)
        policy_obs = _normalise_obs(vecnormalize, aug_obs)
        direction = get_current_direction(env)
        action, _ = model.predict(policy_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)

        raw_observations.append(raw_obs.copy())
        observations.append(aug_obs.copy())
        policy_observations.append(policy_obs.copy())
        actions.append(action.copy())
        directions.append(direction.copy())
        current_goals.append(env.current_goal)
        subgoal_indices.append(env._goal_idx)
        current_cells.append(env._current_cell())
        torso_xy.append(env._get_torso_xy())
        qpos.append(env._data.qpos.copy())
        qvel.append(env._data.qvel.copy())
        ctrl.append(env._data.ctrl.copy())

        if args.quiet_env_debug:
            with contextlib.redirect_stdout(io.StringIO()):
                raw_obs, reward, done, info = env.step(action)
        else:
            raw_obs, reward, done, info = env.step(action)
        final_info = info

        rewards.append(float(reward))
        dones.append(bool(done))
        goal_reached.append(bool(info.get("goal_reached", False)))
        wrong = bool(info.get("wrong_color", False))
        wrong_color.append(wrong)
        hit_wall.append(bool(info.get("hit_wall", False)))
        fell.append(bool(info.get("fell", False)))
        timeout.append(bool(info.get("timeout", False)))

        wrong_color_any = wrong_color_any or wrong
        goals_reached += int(info.get("goal_reached", False))

    final_raw_obs = raw_obs.copy()
    final_obs = _make_aug_obs(env, final_raw_obs)
    final_policy_obs = _normalise_obs(vecnormalize, final_obs)
    final_cell = np.asarray(env._current_cell(), dtype=np.int16)

    clean_success = (
        bool(final_info.get("success", False))
        and goals_reached == len(task)
        and not wrong_color_any
    )

    return dict(
        task=np.asarray(task, dtype=np.int16),
        seed=np.int64(seed),
        clean_path_possible=np.bool_(clean_possible),
        clean_success=np.bool_(clean_success),
        env_success=np.bool_(bool(final_info.get("success", False))),
        wrong_color_any=np.bool_(wrong_color_any),
        goals_reached=np.int16(goals_reached),
        length=np.int32(len(actions)),
        object_cells=object_cells,
        wall_cells=wall_cells,
        start_cell=np.asarray(env._start_cell, dtype=np.int16),
        final_cell=final_cell,
        raw_observations=np.asarray(raw_observations, dtype=np.float32),
        observations=np.asarray(observations, dtype=np.float32),
        policy_observations=np.asarray(policy_observations, dtype=np.float32),
        final_raw_observation=final_raw_obs.astype(np.float32),
        final_observation=final_obs.astype(np.float32),
        final_policy_observation=final_policy_obs.astype(np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        directions=np.asarray(directions, dtype=np.float32),
        current_goals=np.asarray(current_goals, dtype=np.int16),
        subgoal_indices=np.asarray(subgoal_indices, dtype=np.int16),
        current_cells=np.asarray(current_cells, dtype=np.int16),
        torso_xy=np.asarray(torso_xy, dtype=np.float32),
        qpos=np.asarray(qpos, dtype=np.float32),
        qvel=np.asarray(qvel, dtype=np.float32),
        ctrl=np.asarray(ctrl, dtype=np.float32),
        goal_reached=np.asarray(goal_reached, dtype=np.bool_),
        wrong_color=np.asarray(wrong_color, dtype=np.bool_),
        hit_wall=np.asarray(hit_wall, dtype=np.bool_),
        fell=np.asarray(fell, dtype=np.bool_),
        timeout=np.asarray(timeout, dtype=np.bool_),
    )


def _write_trajectory(group: h5py.Group, traj: dict) -> None:
    for key, value in traj.items():
        if np.isscalar(value):
            group.attrs[key] = value
        else:
            group.create_dataset(key, data=value, compression="gzip")


def _write_progress(path: Path, progress: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(progress, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _iter_tasks(args: argparse.Namespace) -> Iterable[tuple[int, ...]]:
    if args.task_index is not None:
        yield tuple(ANT_PRETRAINING_TASKS[args.task_index])
        return
    for task in ANT_PRETRAINING_TASKS:
        yield tuple(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vecnormalize", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/ant_pinpad_test4_long_sweep"))
    parser.add_argument(
        "--checkpoint-pattern",
        default="*bs2048*/*/rl_model_*_steps.zip",
        help="Glob relative to --checkpoint-root used when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--checkpoint-contains",
        default="bs2048",
        help="Substring that must appear in auto-selected checkpoint paths.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/datasets/ant_pinpad_w2_bs2048_expert.h5"))
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-attempts-per-task", type=int, default=1000)
    parser.add_argument(
        "--max-sampled-layouts-per-task",
        type=int,
        help="Safety cap for sampled layouts, including impossible layouts. Defaults to 20x max policy attempts.",
    )
    parser.add_argument("--task-index", type=int, choices=range(len(ANT_PRETRAINING_TASKS)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-walls", type=int, default=2)
    parser.add_argument("--require-clean-path-possible", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet-env-debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subgoal-reward", type=float, default=2.0)
    parser.add_argument("--survival-reward", type=float, default=0.1)
    parser.add_argument("--intrinsic-scale", type=float, default=12.0)
    parser.add_argument("--posture-penalty-coef", type=float, default=10.0)
    parser.add_argument("--posture-penalty-threshold", type=float, default=0.7)
    parser.add_argument("--fall-penalty", type=float, default=10.0)
    args = parser.parse_args()
    if args.max_sampled_layouts_per_task is None:
        args.max_sampled_layouts_per_task = max(args.max_attempts_per_task * 20, args.max_attempts_per_task)

    checkpoint = args.checkpoint or _find_latest_checkpoint(
        args.checkpoint_root,
        args.checkpoint_pattern,
        args.checkpoint_contains,
    )
    vecnormalize_path = args.vecnormalize if args.vecnormalize is not None else _vecnormalize_for(checkpoint)
    print(f"[collector] checkpoint:   {checkpoint}")
    print(f"[collector] vecnormalize: {vecnormalize_path}")
    print(f"[collector] output:       {args.output}")
    progress_path = args.progress_output or args.output.with_suffix(args.output.suffix + ".progress.json")
    print(f"[collector] progress:     {progress_path}")

    import sbx

    model = sbx.PPO.load(str(checkpoint))
    vecnormalize = _load_vecnormalize(vecnormalize_path, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    total_policy_attempts = 0
    total_sampled_layouts = 0

    with h5py.File(args.output, "w") as h5:
        h5.attrs["checkpoint"] = str(checkpoint)
        h5.attrs["vecnormalize"] = "" if vecnormalize_path is None else str(vecnormalize_path)
        h5.attrs["n_walls"] = args.n_walls
        h5.attrs["success_definition"] = "env_success and all 3 subgoals reached and no wrong_color"
        h5.attrs["env_config"] = json.dumps(
            dict(
                n_walls=args.n_walls,
                subgoal_reward=args.subgoal_reward,
                survival_reward=args.survival_reward,
                intrinsic_scale=args.intrinsic_scale,
                posture_penalty_coef=args.posture_penalty_coef,
                posture_penalty_threshold=args.posture_penalty_threshold,
                fall_penalty=args.fall_penalty,
            ),
            sort_keys=True,
        )

        for task_idx, task in enumerate(_iter_tasks(args)):
            task_group = h5.create_group(f"task_{task_idx:02d}")
            task_group.attrs["task"] = np.asarray(task, dtype=np.int16)
            kept = 0
            sampled_layouts = 0
            policy_attempts = 0
            rejected_impossible = 0
            rejected_policy = 0

            while kept < args.episodes_per_task:
                if policy_attempts >= args.max_attempts_per_task:
                    raise RuntimeError(
                        f"Task {task} collected {kept}/{args.episodes_per_task} "
                        f"after {policy_attempts} policy attempts "
                        f"({sampled_layouts} sampled layouts, {rejected_impossible} impossible)."
                    )
                if sampled_layouts >= args.max_sampled_layouts_per_task:
                    raise RuntimeError(
                        f"Task {task} collected {kept}/{args.episodes_per_task} "
                        f"after sampling {sampled_layouts} layouts "
                        f"({policy_attempts} policy attempts, {rejected_impossible} impossible)."
                    )

                seed = args.seed + task_idx * 1_000_000 + sampled_layouts
                traj = collect_one_attempt(model, vecnormalize, task, seed, args)
                sampled_layouts += 1
                total_sampled_layouts += 1

                if args.require_clean_path_possible and not bool(traj["clean_path_possible"]):
                    rejected_impossible += 1
                    continue

                policy_attempts += 1
                total_policy_attempts += 1

                if not bool(traj["clean_success"]):
                    rejected_policy += 1
                    continue

                traj_group = task_group.create_group(f"traj_{kept:05d}")
                _write_trajectory(traj_group, traj)
                kept += 1
                total_kept += 1
                h5.flush()
                _write_progress(
                    progress_path,
                    dict(
                        output=str(args.output),
                        checkpoint=str(checkpoint),
                        current_task=task_idx,
                        current_task_sampled_layouts=sampled_layouts,
                        current_task_policy_attempts=policy_attempts,
                        current_task_kept=kept,
                        current_task_rejected_impossible=rejected_impossible,
                        current_task_rejected_policy=rejected_policy,
                        episodes_per_task=args.episodes_per_task,
                        total_sampled_layouts=total_sampled_layouts,
                        total_policy_attempts=total_policy_attempts,
                        total_kept=total_kept,
                    ),
                )

                print(
                    f"[collector] task={task_idx:02d} kept={kept}/{args.episodes_per_task} "
                    f"sampled={sampled_layouts} policy_attempts={policy_attempts} "
                    f"impossible={rejected_impossible} policy_fail={rejected_policy} "
                    f"len={int(traj['length'])}"
                )

            task_group.attrs["sampled_layouts"] = sampled_layouts
            task_group.attrs["policy_attempts"] = policy_attempts
            task_group.attrs["kept"] = kept
            task_group.attrs["rejected_impossible"] = rejected_impossible
            task_group.attrs["rejected_policy"] = rejected_policy
            h5.flush()
            print(
                f"[collector] task={task_idx:02d} done kept={kept} "
                f"sampled={sampled_layouts} policy_attempts={policy_attempts} "
                f"impossible={rejected_impossible} policy_fail={rejected_policy}"
            )

        h5.attrs["total_kept"] = total_kept
        h5.attrs["total_sampled_layouts"] = total_sampled_layouts
        h5.attrs["total_policy_attempts"] = total_policy_attempts

    print(
        f"[collector] wrote {total_kept} trajectories from "
        f"{total_sampled_layouts} sampled layouts and {total_policy_attempts} policy attempts"
    )


if __name__ == "__main__":
    main()
