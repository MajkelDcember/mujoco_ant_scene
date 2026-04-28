"""
Ant-pinpad environment.

Spec (Appendix A.2 of the paper):
  - G×G grid of cells (default 4×4), each cell_size × cell_size MuJoCo units
  - O coloured cells + W wall cells, randomly placed each episode
  - Agent: MuJoCo 'ant' quadruped (8 joints, continuous torque control)
  - Action: 8D continuous vector (joint torques), clipped to [-1, 1]
  - Observation:
      - ant proprioception (symlog-scaled)
      - normalised global x,y ant coordinate in [-1, 1]
      - relative positions of coloured cells and walls w.r.t. ant
      - local coordinate of ant within current cell
  - Reward: +1 on full task completion (sparse)
  - Termination:
      - task complete
      - wrong coloured cell visited
      - torso z outside [0.2, 1.0]
      - wall cell entered
      - T steps elapsed

Expert policy (Appendix C.1.1):
  - Trained PPO agent with:
      * Augmented obs: shortest-path direction vector for every grid cell
      * Intrinsic reward: dot(velocity, direction)
  - After training, rollout using mean of Gaussian policy (no sampling).

Note: MuJoCo (dm_control or mujoco-py) must be installed.
We use the `mujoco` Python bindings (pip install mujoco).
The ant XML is taken from dm_control / D4RL / standard MuJoCo distribution.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# ---------------------------------------------------------------------------
# Try to import MuJoCo; provide informative error if missing
# ---------------------------------------------------------------------------
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

# ---------------------------------------------------------------------------
# Ant XML (standard MuJoCo ant, matches D4RL / dm_control)
# Embed a minimal version so the env is self-contained.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Colour palette for the 4 ant-pinpad subgoal colours (RGBA)
# Matches the paper figure: red, blue, green, yellow
# ---------------------------------------------------------------------------
TILE_COLORS = {
    0: (0.85, 0.20, 0.15, 1.0),   # red
    1: (0.15, 0.45, 0.85, 1.0),   # blue
    2: (0.10, 0.70, 0.30, 1.0),   # green
    3: (0.90, 0.75, 0.05, 1.0),   # yellow
}
WALL_COLOR_RGBA = (0.35, 0.25, 0.15, 1.0)   # dark brown, like paper
FLOOR_RGBA      = (0.82, 0.71, 0.55, 1.0)   # sandy floor (paper figure)
CARDINAL_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# Overhead-camera viewer perspective (y increases = Up on screen):
#   index 0  row-1  →  Down  (lower y, lower row)
#   index 1  row+1  →  Up    (higher y, higher row)
#   index 2  col-1  →  Left
#   index 3  col+1  →  Right
CARDINAL_WORLD_VECS = np.array([
    [0.0, -1.0],  # Down  (row-1 → lower y)
    [0.0,  1.0],  # Up    (row+1 → higher y)
    [-1.0, 0.0],  # Left  (col-1 → lower x)
    [1.0,  0.0],  # Right (col+1 → higher x)
], dtype=np.float32)


def build_ant_xml(
    object_cells: Dict[int, Tuple[int, int]],   # colour → (row, col)
    wall_cells:   List[Tuple[int, int]],
    ant_start_xy: Tuple[float, float],
    grid_size:    int   = 4,
    cell_size:    float = 2.0,
    wall_height:  float = 1.0,
    tile_thickness: float = 0.02,
) -> str:
    """
    Build a MuJoCo XML string with:
      - Coloured floor tiles for each subgoal cell
      - Wall boxes for each wall cell
      - An overhead camera looking straight down at the grid centre
      - A side/perspective camera (like the paper figure)
      - The standard ant body

    Called on every episode reset with the new layout.
    """
    G  = grid_size
    cs = cell_size
    ox = -(G * cs) / 2.0
    oy = -(G * cs) / 2.0

    # Grid centre
    cx = 0.0
    cy = 0.0

    # Overhead camera height
    cam_h = G * cs * 1.4

    # Build tile geoms (coloured subgoal cells)
    tile_geoms = []
    for color, (row, col) in object_cells.items():
        tx = ox + (col + 0.5) * cs
        ty = oy + (row + 0.5) * cs
        r, g, b, a = TILE_COLORS.get(color, (0.8, 0.8, 0.8, 1.0))
        tile_geoms.append(
            f'    <geom name="tile_{color}" type="box" '
            f'pos="{tx:.4f} {ty:.4f} {tile_thickness/2:.4f}" '
            f'size="{cs/2*0.98:.4f} {cs/2*0.98:.4f} {tile_thickness/2:.4f}" '
            f'rgba="{r} {g} {b} {a}" contype="0" conaffinity="0"/>'
        )

    # Build internal wall geoms
    wall_geoms = []
    for i, (row, col) in enumerate(wall_cells):
        wx = ox + (col + 0.5) * cs
        wy = oy + (row + 0.5) * cs
        r, g, b, a = WALL_COLOR_RGBA
        wall_geoms.append(
            f'    <geom name="wall_{i}" type="box" '
            f'pos="{wx:.4f} {wy:.4f} {wall_height/2:.4f}" '
            f'size="{cs/2*0.95:.4f} {cs/2*0.95:.4f} {wall_height/2:.4f}" '
            f'rgba="{r} {g} {b} {a}" contype="1" conaffinity="1"/>'
        )

    # Build arena perimeter walls (like the paper figure)
    arena_half  = G * cs / 2.0
    pwall_t     = 0.3           # wall thickness
    pwall_h     = wall_height * 1.5
    wr, wg, wb, wa = WALL_COLOR_RGBA
    perimeter_walls = f"""
    <geom name="pwall_south" type="box"
          pos="{cx:.4f} {oy - pwall_t/2:.4f} {pwall_h/2:.4f}"
          size="{arena_half + pwall_t:.4f} {pwall_t/2:.4f} {pwall_h/2:.4f}"
          rgba="{wr} {wg} {wb} {wa}" contype="1" conaffinity="1"/>
    <geom name="pwall_north" type="box"
          pos="{cx:.4f} {-oy + pwall_t/2:.4f} {pwall_h/2:.4f}"
          size="{arena_half + pwall_t:.4f} {pwall_t/2:.4f} {pwall_h/2:.4f}"
          rgba="{wr} {wg} {wb} {wa}" contype="1" conaffinity="1"/>
    <geom name="pwall_west" type="box"
          pos="{ox - pwall_t/2:.4f} {cy:.4f} {pwall_h/2:.4f}"
          size="{pwall_t/2:.4f} {arena_half:.4f} {pwall_h/2:.4f}"
          rgba="{wr} {wg} {wb} {wa}" contype="1" conaffinity="1"/>
    <geom name="pwall_east" type="box"
          pos="{-ox + pwall_t/2:.4f} {cy:.4f} {pwall_h/2:.4f}"
          size="{pwall_t/2:.4f} {arena_half:.4f} {pwall_h/2:.4f}"
          rgba="{wr} {wg} {wb} {wa}" contype="1" conaffinity="1"/>"""

    fr, fg, fb, fa = FLOOR_RGBA
    tiles_str = "\n".join(tile_geoms)
    walls_str = "\n".join(wall_geoms)

    ax, ay = ant_start_xy

    return f"""
<mujoco model="ant_pinpad">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="3" density="5.0" friction="1.5 0.1 0.1"
          margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <worldbody>
    <!-- Lighting -->
    <light name="main_light" pos="{cx} {cy} {cam_h*0.8:.2f}"
           dir="0 0 -1" directional="true"
           diffuse="0.9 0.9 0.9" specular="0.1 0.1 0.1" castshadow="true"/>
    <light name="fill_light" pos="{cx - G*cs} {cy} {cam_h*0.5:.2f}"
           dir="0.3 0 -0.7" directional="true"
           diffuse="0.4 0.4 0.4" specular="0 0 0" castshadow="false"/>

    <!-- Floor -->
    <geom name="floor" type="plane"
          pos="0 0 0" size="{G*cs*2:.1f} {G*cs*2:.1f} 0.1"
          rgba="{fr} {fg} {fb} {fa}" friction="1.5 0.1 0.1"/>

    <!-- Coloured subgoal tiles -->
{tiles_str}

    <!-- Internal wall obstacles -->
{walls_str}

    <!-- Arena perimeter walls (visual + collision) -->
{perimeter_walls}

    <!-- Overhead camera (straight down, always correct) -->
    <camera name="overhead"
            pos="{cx:.4f} {cy:.4f} {cam_h:.4f}"
            xyaxes="1 0 0 0 1 0"/>

    <!-- Perspective camera: static 45° isometric, centred on grid -->
    <camera name="perspective"
            pos="0.0 -8.0 8.0"
            xyaxes="1 0 0 0 0.707 0.707"/>

    <!-- Ant body -->
    <body name="torso" pos="{ax:.4f} {ay:.4f} 0.75">
      <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"
            rgba="0.7 0.7 0.7 1"/>
      <joint armature="0" damping="0" limited="false" margin="0.01"
             name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0 0 0 0.2 0.2 0" name="aux_1_geom" size="0.08" type="capsule"/>
        <body name="aux_1" pos="0.2 0.2 0">
          <joint axis="0 0 1" name="hip_1" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 0.2 0.2 0" name="left_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 0.2 0">
            <joint axis="-1 1 0" name="ankle_1" pos="0 0 0" range="30 70" type="hinge"/>
            <geom fromto="0 0 0 0.4 0.4 0" name="left_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0 0 0 -0.2 0.2 0" name="aux_2_geom" size="0.08" type="capsule"/>
        <body name="aux_2" pos="-0.2 0.2 0">
          <joint axis="0 0 1" name="hip_2" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 -0.2 0.2 0" name="right_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 0.2 0">
            <joint axis="1 1 0" name="ankle_2" pos="0 0 0" range="-70 -30" type="hinge"/>
            <geom fromto="0 0 0 -0.4 0.4 0" name="right_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="back_leg" pos="0 0 0">
        <geom fromto="0 0 0 -0.2 -0.2 0" name="aux_3_geom" size="0.08" type="capsule"/>
        <body name="aux_3" pos="-0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_3" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 -0.2 -0.2 0" name="back_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 -0.2 0">
            <joint axis="-1 1 0" name="ankle_3" pos="0 0 0" range="-70 -30" type="hinge"/>
            <geom fromto="0 0 0 -0.4 -0.4 0" name="back_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0 0 0 0.2 -0.2 0" name="aux_4_geom" size="0.08" type="capsule"/>
        <body name="aux_4" pos="0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_4" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 0.2 -0.2 0" name="right_back_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 -0.2 0">
            <joint axis="1 1 0" name="ankle_4" pos="0 0 0" range="30 70" type="hinge"/>
            <geom fromto="0 0 0 0.4 -0.4 0" name="right_back_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1 1" joint="hip_4"   name="hip_4"   gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1 1" joint="ankle_4" name="ankle_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1 1" joint="hip_1"   name="hip_1"   gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1 1" joint="ankle_1" name="ankle_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1 1" joint="hip_2"   name="hip_2"   gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1 1" joint="ankle_2" name="ankle_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1 1" joint="hip_3"   name="hip_3"   gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1 1" joint="ankle_3" name="ankle_3" gear="150"/>
  </actuator>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Helper: symlog transform (paper uses this for proprioception)
# ---------------------------------------------------------------------------

def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log(np.abs(x) + 1.0)


# ---------------------------------------------------------------------------
# Grid layout helper
# ---------------------------------------------------------------------------

class GridLayout:
    """Maps grid (row, col) indices to MuJoCo world (x, y) coordinates."""

    def __init__(self, grid_size: int, cell_size: float = 2.0):
        self.G    = grid_size
        self.cs   = cell_size
        # World origin at bottom-left corner of grid
        self.origin_x = -(grid_size * cell_size) / 2.0
        self.origin_y = -(grid_size * cell_size) / 2.0

    def cell_center(self, row: int, col: int) -> Tuple[float, float]:
        """Return (x, y) world coordinates of cell center."""
        x = self.origin_x + (col + 0.5) * self.cs
        y = self.origin_y + (row + 0.5) * self.cs
        return x, y

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Return (row, col) for world (x, y); clamps to grid bounds."""
        col = int((x - self.origin_x) / self.cs)
        row = int((y - self.origin_y) / self.cs)
        row = max(0, min(self.G - 1, row))
        col = max(0, min(self.G - 1, col))
        return row, col

    def local_coord(self, x: float, y: float) -> Tuple[float, float]:
        """Return local (dx, dy) within current cell, normalised to [-0.5, 0.5]."""
        row, col = self.world_to_cell(x, y)
        cx, cy   = self.cell_center(row, col)
        return (x - cx) / self.cs, (y - cy) / self.cs

    def world_extent(self) -> Tuple[float, float, float, float]:
        """xmin, xmax, ymin, ymax of the grid."""
        xmin = self.origin_x
        xmax = self.origin_x + self.G * self.cs
        ymin = self.origin_y
        ymax = self.origin_y + self.G * self.cs
        return xmin, xmax, ymin, ymax


# ---------------------------------------------------------------------------
# Ant-pinpad environment
# ---------------------------------------------------------------------------

class AntPinpad:
    """
    Ant-pinpad environment.

    Parameters
    ----------
    task : tuple[int, ...]
        Ordered sequence of colour IDs to visit.
    grid_size : int
    n_objects : int
    n_walls : int
    max_steps : int
    cell_size : float
        Side length of each grid cell in MuJoCo world units.
    torso_z_min / torso_z_max : float
        Valid torso height range.
    seed : int | None
    """

    N_PROPRIOCEPTION = 27  # standard ant: qpos(15) - root_xy(2) + qvel(14) = 27
    # qpos: free joint 7 (xyz + quat) + 8 joint angles = 15
    # qvel: free joint 6 + 8 joint vels = 14
    # We exclude the global x,y from qpos (handled separately), giving:
    # 1 (z) + 4 (quat) + 8 (joint angles) + 6 (root vel) + 8 (joint vels) = 27

    def __init__(
        self,
        task:         Tuple[int, ...],
        grid_size:    int   = 4,
        n_objects:    int   = 4,
        n_walls:      int   = 1,
        max_steps:    int   = 500,
        cell_size:    float = 2.0,
        torso_z_min:  float = 0.2,
        torso_z_max:  float = 2.0,
        random_yaw:   bool  = True,
        start_pos_noise_scale:   float = 0.1,
        exploration_bonus_scale: float = 0.0,
        subgoal_reward:          float = 1.0,
        wrong_color_kills:       bool  = True,
        wrong_color_penalty:     float = -1.0,
        fall_penalty:            float = -1.0,
        wall_penalty:            float = -0.5,
        avoid_nontask_colors:    bool  = False,
        seed:         Optional[int] = None,
    ):
        if not HAS_MUJOCO:
            raise ImportError(
                "mujoco Python package not found. Install with: pip install mujoco"
            )

        self.task        = tuple(task)
        self.G           = grid_size
        self.O           = n_objects
        self.W           = n_walls
        self.max_steps   = max_steps
        self.cell_size   = cell_size
        self.torso_z_min = torso_z_min
        self.torso_z_max = torso_z_max
        self.random_yaw  = random_yaw
        self.start_pos_noise_scale   = start_pos_noise_scale
        self.exploration_bonus_scale = exploration_bonus_scale
        self.subgoal_reward      = subgoal_reward
        self.wrong_color_kills   = wrong_color_kills
        self.wrong_color_penalty = wrong_color_penalty
        self.fall_penalty          = fall_penalty
        self.wall_penalty          = wall_penalty
        self.avoid_nontask_colors  = avoid_nontask_colors
        self.rng         = np.random.default_rng(seed)

        self.layout = GridLayout(grid_size, cell_size)

        # Observation dimension:
        # proprioception (27) + global xy (2) + relative positions of O objects (O*2)
        # + relative positions of W walls (W*2) + local cell coord (2)
        self.obs_dim = (
            self.N_PROPRIOCEPTION
            + 2                # global x,y normalised
            + self.O * 2       # relative positions of objects
            + self.W * 2       # relative positions of walls
            + 2                # local cell coord
        )

        # Layout state (populated in reset)
        self._object_cells: Dict[int, Tuple[int, int]] = {}
        self._wall_cells:   List[Tuple[int, int]] = []
        self._start_cell:   Tuple[int, int] = (0, 0)

        # MuJoCo model/data — built on first reset()
        self._model = None
        self._data  = None

        # Episode state
        self._step     = 0
        self._goal_idx = 0
        self._prev_cell: Optional[Tuple[int, int]] = None
        self._visited_cells: set = set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_grid_layout(self) -> None:
        """Randomly place objects, walls, and agent start cell."""
        G = self.G
        all_cells = [(r, c) for r in range(G) for c in range(G)]
        chosen_idx = self.rng.choice(
            len(all_cells), size=self.O + self.W + 1, replace=False
        )
        chosen = [all_cells[i] for i in chosen_idx]

        self._object_cells = {color: chosen[color] for color in range(self.O)}
        self._wall_cells   = chosen[self.O : self.O + self.W]
        self._start_cell   = chosen[self.O + self.W]

    def _init_ant_pose(self, ant_start_xy: Tuple[float, float]) -> None:
        """Place ant at given world (x, y) with random yaw + small noise."""
        x, y = ant_start_xy

        yaw  = self.rng.uniform(0, 2 * np.pi) if self.random_yaw else 0.0
        quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

        # Set free joint: [x, y, z, qw, qx, qy, qz]
        self._data.qpos[:7] = [x, y, 0.75, quat[0], quat[1], quat[2], quat[3]]

        # Small random joint angles
        self._data.qpos[7:] = self.rng.uniform(-0.1, 0.1, size=len(self._data.qpos) - 7)

        # Small random velocities
        self._data.qvel[:] = self.rng.uniform(-0.1, 0.1, size=len(self._data.qvel))

        mujoco.mj_forward(self._model, self._data)

    def _get_torso_xy(self) -> Tuple[float, float]:
        return float(self._data.qpos[0]), float(self._data.qpos[1])

    def _get_torso_z(self) -> float:
        return float(self._data.qpos[2])

    def _torso_up_from_quat(self) -> np.ndarray:
        """Return the body +Z axis in world frame (the 'up' direction of the torso)."""
        qw, qx, qy, qz = [float(v) for v in self._data.qpos[3:7]]
        return np.array([
            2.0 * (qx*qz + qw*qy),
            2.0 * (qy*qz - qw*qx),
            1.0 - 2.0 * (qx*qx + qy*qy),
        ], dtype=np.float64)

    def _get_proprioception(self) -> np.ndarray:
        """27-dim proprioception (symlog-scaled), excluding global x,y."""
        # qpos: skip x,y (indices 0,1); take z, quat, joint angles
        pos_part = self._data.qpos[2:]   # z(1) + quat(4) + joints(8) = 13
        vel_part = self._data.qvel[:]    # root_vel(6) + joint_vels(8) = 14
        raw = np.concatenate([pos_part, vel_part])   # 27
        return symlog(raw).astype(np.float32)

    def _make_obs(self) -> np.ndarray:
        """Build observation vector."""
        x, y = self._get_torso_xy()
        xmin, xmax, ymin, ymax = self.layout.world_extent()

        # Normalised global coordinates
        gx = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
        gy = 2.0 * (y - ymin) / (ymax - ymin) - 1.0

        # Relative positions of objects (in world units, from ant's perspective)
        obj_rel = []
        for color in range(self.O):
            row, col = self._object_cells[color]
            ox, oy   = self.layout.cell_center(row, col)
            obj_rel.extend([(ox - x) / (self.G * self.cell_size),
                            (oy - y) / (self.G * self.cell_size)])

        # Relative positions of walls
        wall_rel = []
        for (row, col) in self._wall_cells:
            wx, wy = self.layout.cell_center(row, col)
            wall_rel.extend([(wx - x) / (self.G * self.cell_size),
                             (wy - y) / (self.G * self.cell_size)])

        # Local coord within current cell
        lx, ly = self.layout.local_coord(x, y)

        prop = self._get_proprioception()
        obs  = np.concatenate([
            prop,
            [gx, gy],
            obj_rel,
            wall_rel,
            [lx, ly],
        ]).astype(np.float32)

        return obs

    def _current_cell(self) -> Tuple[int, int]:
        x, y = self._get_torso_xy()
        return self.layout.world_to_cell(x, y)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _spawn_is_valid(self) -> bool:
        """Return True if the current ant pose is safe to start from."""
        # torso must be above the floor threshold
        if self._get_torso_z() < self.torso_z_min:
            return False
        # torso must not be flipped (up_z < 0 means upside-down)
        _, qx, qy, _ = [float(v) for v in self._data.qpos[3:7]]
        up_z = 1.0 - 2.0 * (qx*qx + qy*qy)
        if up_z < 0.0:
            return False
        # no ant geom may be in contact with any wall geom
        wall_geom_ids: set = set()
        for i in range(self.W):
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"wall_{i}")
            if gid >= 0:
                wall_geom_ids.add(gid)
        for name in ("pwall_south", "pwall_north", "pwall_west", "pwall_east"):
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                wall_geom_ids.add(gid)
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            if c.geom1 in wall_geom_ids or c.geom2 in wall_geom_ids:
                return False
        return True

    def reset(self) -> np.ndarray:
        self._step     = 0
        self._goal_idx = 0

        for _ in range(100):
            self._sample_grid_layout()

            row, col  = self._start_cell
            cx, cy    = self.layout.cell_center(row, col)
            noise_mag = self.cell_size * self.start_pos_noise_scale
            noise     = self.rng.uniform(-noise_mag, noise_mag, 2)
            ant_start = (cx + noise[0], cy + noise[1])

            xml = build_ant_xml(
                object_cells = self._object_cells,
                wall_cells   = self._wall_cells,
                ant_start_xy = ant_start,
                grid_size    = self.G,
                cell_size    = self.cell_size,
            )
            self._model = mujoco.MjModel.from_xml_string(xml)
            self._data  = mujoco.MjData(self._model)

            # _init_ant_pose ends with mj_forward, so contacts are populated
            self._init_ant_pose(ant_start)

            if self._spawn_is_valid():
                self._prev_cell     = self._current_cell()
                self._visited_cells = {self._prev_cell}
                return self._make_obs()

        # Last resort: place exactly at cell centre with no noise and a neutral pose
        self._sample_grid_layout()
        row, col  = self._start_cell
        cx, cy    = self.layout.cell_center(row, col)
        ant_start = (cx, cy)
        xml = build_ant_xml(
            object_cells = self._object_cells,
            wall_cells   = self._wall_cells,
            ant_start_xy = ant_start,
            grid_size    = self.G,
            cell_size    = self.cell_size,
        )
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data  = mujoco.MjData(self._model)
        self._data.qpos[:7] = [cx, cy, 0.75, 1.0, 0.0, 0.0, 0.0]
        self._data.qpos[7:] = 0.0
        self._data.qvel[:]  = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._prev_cell     = self._current_cell()
        self._visited_cells = {self._prev_cell}
        return self._make_obs()

    def render(
        self,
        camera:  str = "perspective",   # "perspective" or "overhead"
        width:   int = 480,
        height:  int = 480,
    ) -> np.ndarray:
        """
        Render current state using MuJoCo's offscreen renderer.
        Returns HxWx3 uint8 RGB array.

        Requires EGL or osmesa on the cluster:
            export MUJOCO_GL=egl      # GPU nodes (H100)
            export MUJOCO_GL=osmesa   # CPU-only nodes
        """
        renderer = mujoco.Renderer(self._model, height=height, width=width)
        renderer.update_scene(self._data, camera=camera)
        frame = renderer.render()
        renderer.close()
        return frame

    @property
    def current_goal(self) -> int:
        if self._goal_idx < len(self.task):
            return self.task[self._goal_idx]
        return -1

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Parameters
        ----------
        action : (8,) float array, joint torques in [-1, 1]
        """
        action = np.clip(action, -1.0, 1.0)
        self._data.ctrl[:] = action

        # Simulate one control step (default: 5 physics steps per control step)
        n_substeps = 5
        for _ in range(n_substeps):
            mujoco.mj_step(self._model, self._data)

        self._step += 1

        reward = 0.0
        done   = False
        info   = dict(
            success=False,
            wrong_color=False,
            timeout=False,
            fell=False,
            hit_wall=False,
            goal_reached=False,
            subgoal_index=self._goal_idx,
            abstract_action=self._goal_idx,
        )

        # ── debug helpers ──────────────────────────────────────────────────────
        def _up_vector():
            qw, qx, qy, qz = [float(v) for v in self._data.qpos[3:7]]
            return np.array([
                2.0 * (qx*qz + qw*qy),
                2.0 * (qy*qz - qw*qx),
                1.0 - 2.0 * (qx*qx + qy*qy),
            ])

        def _debug_print(trigger: str, obs: np.ndarray) -> None:
            x = float(self._data.qpos[0])
            y = float(self._data.qpos[1])
            z = float(self._data.qpos[2])
            up = _up_vector()
            try:
                cell = self._current_cell() if (np.isfinite(x) and np.isfinite(y)) else "NaN_xy"
            except Exception:
                cell = "error"
            print(
                f"\n[ANT DEBUG TERMINATION]\n"
                f"  step={self._step}  trigger={trigger}\n"
                f"  xyz=({x:.6f}, {y:.6f}, {z:.6f})\n"
                f"  up_vector=({up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f})\n"
                f"  current_cell={cell}  goal_idx={self._goal_idx}/{len(self.task)}  task={self.task}\n"
                f"  nan_qpos={bool(np.isnan(self._data.qpos).any())}  "
                f"nan_qvel={bool(np.isnan(self._data.qvel).any())}  "
                f"nan_ctrl={bool(np.isnan(self._data.ctrl).any())}  "
                f"nan_obs={bool(np.isnan(obs).any())}\n"
                f"  qvel_norm={float(np.linalg.norm(self._data.qvel)):.4f}\n"
                f"  ctrl={np.array2string(self._data.ctrl.copy(), precision=3, suppress_small=False)}\n"
                f"{'─'*56}"
            )
        # ── end debug helpers ──────────────────────────────────────────────────

        # Catch NaN state before any normal termination logic
        if np.isnan(self._data.qpos).any() or np.isnan(self._data.qvel).any() or np.isnan(self._data.ctrl).any():
            obs = self._make_obs()
            info['nan_state'] = True
            done = True
            _debug_print("nan_state", obs)
            return obs, reward, done, info

        # Check torso height / orientation
        z     = self._get_torso_z()
        up_z  = _up_vector()[2]
        if z < self.torso_z_min or up_z < -0.2:
            done = True
            info['fell'] = True
            reward += self.fall_penalty
            obs = self._make_obs()
            _debug_print("fell", obs)
            return obs, reward, done, info

        # Check cell transitions
        curr_cell = self._current_cell()
        prev_cell = self._prev_cell
        entered_new_cell = (curr_cell != prev_cell)

        if entered_new_cell:
            # Check wall
            if curr_cell in self._wall_cells:
                done = True
                info['hit_wall'] = True
                reward += self.wall_penalty
                self._prev_cell = curr_cell
                obs = self._make_obs()
                _debug_print("hit_wall", obs)
                return obs, reward, done, info

            # The paper uses sparse task reward; keep any exploration shaping off
            # unless it is explicitly requested.
            if curr_cell not in self._visited_cells:
                self._visited_cells.add(curr_cell)
                is_wrong = any(
                    cell == curr_cell
                    and color in self.task
                    and (self._goal_idx >= len(self.task) or color != self.task[self._goal_idx])
                    for color, cell in self._object_cells.items()
                )
                if not is_wrong and self.exploration_bonus_scale > 0.0:
                    reward += self.exploration_bonus_scale

            # Check coloured cells
            for color, cell in self._object_cells.items():
                if curr_cell == cell:
                    if (self._goal_idx < len(self.task)
                            and color == self.task[self._goal_idx]):
                        self._goal_idx += 1
                        info['goal_reached']  = True
                        info['subgoal_index'] = self._goal_idx - 1
                        reward += self.subgoal_reward
                        if self._goal_idx == len(self.task):
                            reward += 1.0   # completion bonus on top of final subgoal bonus
                            done   = True
                            info['success'] = True
                            obs = self._make_obs()
                            _debug_print("success", obs)
                            return obs, reward, done, info
                    elif color in self.task or (self.avoid_nontask_colors and color not in self.task):
                        reward += self.wrong_color_penalty
                        info['wrong_color'] = True
                        if self.wrong_color_kills:
                            done = True
                            obs = self._make_obs()
                            _debug_print("wrong_color", obs)
                            return obs, reward, done, info
                    break

        self._prev_cell = curr_cell

        if not done and self._step >= self.max_steps:
            done = True
            info['timeout'] = True
            obs = self._make_obs()
            _debug_print("timeout", obs)
            return obs, reward, done, info

        info['abstract_action'] = min(self._goal_idx, len(self.task) - 1)
        obs = self._make_obs()

        # Catch NaN obs even when state looked finite
        if np.isnan(obs).any():
            info['nan_obs'] = True
            done = True
            _debug_print("nan_obs", obs)
            return obs, reward, done, info

        return obs, reward, done, info

    @property
    def n_actions(self) -> int:
        return 8

    @property
    def observation_dim(self) -> int:
        return self.obs_dim


# ---------------------------------------------------------------------------
# Shortest-path direction helper (for PPO expert augmented observation)
# ---------------------------------------------------------------------------

def _bfs_grid(
    start:      Tuple[int, int],
    goal:       Tuple[int, int],
    grid_size:  int,
    wall_cells: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """
    Return the (dr, dc) of the first step on BFS shortest path from start to goal
    on the grid graph. Returns None if already at goal or unreachable.
    """
    if start == goal:
        return None
    walls   = set(map(tuple, wall_cells))
    visited = {start}
    queue   = []  # (pos, first_delta)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c  = start
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in walls:
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), (dr, dc)))

    from collections import deque
    dq = deque(queue)
    while dq:
        pos, first_delta = dq.popleft()
        if pos == goal:
            return first_delta
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c   = pos
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in walls:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    dq.append(((nr, nc), first_delta))
    return None


def build_direction_field(
    grid_size: int,
    wall_cells: List[Tuple[int, int]],
    goal_cell: Tuple[int, int],
) -> np.ndarray:
    """
    Build a per-tile cardinal-direction field towards `goal_cell`.

    Returns shape (G, G, 4), where the last dimension is a one-hot over
    [row-1, row+1, col-1, col+1] = [Down, Up, Left, Right] from the
    overhead-camera viewer perspective (y increases = Up on screen).
    Wall tiles and the goal tile receive an all-zero vector.
    """
    field = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
    walls = set(map(tuple, wall_cells))

    for row in range(grid_size):
        for col in range(grid_size):
            cell = (row, col)
            if cell in walls or cell == goal_cell:
                continue
            delta = _bfs_grid(cell, goal_cell, grid_size, wall_cells)
            if delta in CARDINAL_DELTAS:
                field[row, col, CARDINAL_DELTAS.index(delta)] = 1.0

    return field


def render_direction_field_text(
    field: np.ndarray,
    wall_cells: List[Tuple[int, int]],
    goal_cell: Tuple[int, int],
    agent_cell: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Text rendering of a direction field from the overhead-camera viewer perspective.

    Printed with row G-1 at the top (matches overhead camera: y increases = Up).
    Direction arrows:
        ^  Up    (index 1, row+1 → higher y)
        v  Down  (index 0, row-1 → lower y)
        <  Left  (index 2, col-1)
        >  Right (index 3, col+1)

    Cells: G=goal  #=wall  @=agent  .=no path  arrow otherwise
    """
    G = field.shape[0]
    # index→arrow using overhead-viewer convention
    _DIR = ['v', '^', '<', '>']  # [Down, Up, Left, Right] for indices [0,1,2,3]
    walls = set(map(tuple, wall_cells))

    lines = [f"direction field  goal={goal_cell}  (row {G-1} at top = Up)"]
    for r in range(G - 1, -1, -1):     # row G-1 printed first = top of screen
        row_chars = []
        for c in range(G):
            cell = (r, c)
            if agent_cell and cell == agent_cell:
                row_chars.append('@')
            elif cell == goal_cell:
                row_chars.append('G')
            elif cell in walls:
                row_chars.append('#')
            elif field[r, c].sum() == 0:
                row_chars.append('.')
            else:
                row_chars.append(_DIR[int(field[r, c].argmax())])
        lines.append(' '.join(row_chars))
    return '\n'.join(lines)


def get_direction_augmentation(env: AntPinpad) -> np.ndarray:
    """
    Return the 4D cardinal direction for the ant's current tile only.

    This is identical to get_current_direction — a (4,) one-hot
    [up, down, left, right] pointing toward the current subgoal.
    The full G*G*4 field is NOT used as observation input; only the
    current-cell entry matters and the rest would be noise.
    """
    return get_current_direction(env)


def get_current_direction(env: AntPinpad) -> np.ndarray:
    """Return the 4D cardinal direction for the ant's current tile only.

    BFS treats walls AND every non-target coloured cell as impassable, so the
    direction signal never routes the agent through any coloured tile (task or
    non-task).  If the target is only reachable through coloured tiles (rare
    random layout), falls back to walls-only BFS to preserve some signal.
    """
    if env._goal_idx >= len(env.task):
        return np.zeros(4, dtype=np.float32)

    target_color = env.task[env._goal_idx]
    target_cell  = env._object_cells[target_color]
    curr_cell    = env._current_cell()

    # Block walls + every coloured cell that is not the current target.
    # Non-task colours are included so the BFS never routes through any tile that
    # could confuse the agent, regardless of whether it carries a penalty.
    blocked = set(map(tuple, env._wall_cells))
    for color, cell in env._object_cells.items():
        if color != target_color:
            blocked.add(tuple(cell))

    direction = np.zeros(4, dtype=np.float32)
    delta = _bfs_grid(curr_cell, target_cell, env.G, list(blocked))
    if delta in CARDINAL_DELTAS:
        direction[CARDINAL_DELTAS.index(delta)] = 1.0
    elif delta is None:
        # Target unreachable without crossing a wrong-color tile — fall back to
        # walls-only BFS so the agent still has some directional signal.
        delta = _bfs_grid(curr_cell, target_cell, env.G, list(env._wall_cells))
        if delta in CARDINAL_DELTAS:
            direction[CARDINAL_DELTAS.index(delta)] = 1.0
    return direction


# ---------------------------------------------------------------------------
# PPO Expert Trainer
# ---------------------------------------------------------------------------

class AntPPOExpert:
    """
    Trains a PPO agent on the ant-pinpad environment with shaped rewards.

    The augmented observation = env_obs + direction field (G * G * 4).
    Intrinsic reward = dot(xy_velocity, direction_vector).

    Uses a simple MLP Gaussian policy.

    Parameters
    ----------
    tasks : list of task tuples (train on all tasks jointly)
    train_steps : int
    lr : float
    gamma : float
    gae_lambda : float
    clip_eps : float
    entropy_coef : float
    n_envs : int    Number of parallel environments (sequential here)
    rollout_steps : int  Steps per environment per update
    **env_kwargs : passed to AntPinpad
    """

    def __init__(
        self,
        tasks:         List[Tuple[int, ...]],
        train_steps:   int   = 512_000,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        clip_eps:      float = 0.2,
        entropy_coef:  float = 3e-4,
        intrinsic_scale: float = 0.5,
        weight_decay:  float = 0.03,
        rollout_steps: int   = 2048,
        n_epochs:      int   = 10,
        batch_size:    int   = 64,
        hidden_dim:    int   = 256,
        seed:          int   = 0,
        **env_kwargs,
    ):
        if not HAS_MUJOCO:
            raise ImportError("mujoco not found. pip install mujoco")

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("torch not found. pip install torch")

        self.tasks        = tasks
        self.train_steps  = train_steps
        self.lr           = lr
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.intrinsic_scale = intrinsic_scale
        self.rollout_steps = rollout_steps
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.seed         = seed
        self.env_kwargs   = env_kwargs

        # Build a sample env to get obs/act dims
        sample_env = AntPinpad(tasks[0], **env_kwargs, seed=seed)
        self.obs_dim = sample_env.obs_dim + 4   # +4: current-cell direction only
        self.act_dim = sample_env.n_actions
        del sample_env

        import torch.nn as nn

        self.policy = self._build_policy(hidden_dim, nn)
        self.device = "cuda" if _torch_cuda() else "cpu"
        self.policy.to(self.device)

        import torch.optim as optim
        self.optimizer = optim.AdamW(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self._rng = np.random.default_rng(seed)
        self._trained = False

    def _build_policy(self, hidden_dim: int, nn):
        """Simple MLP Gaussian policy + value head."""
        import torch
        import torch.nn as tnn

        class GaussianPolicy(tnn.Module):
            def __init__(self, obs_dim, act_dim, hidden):
                super().__init__()
                self.shared = tnn.Sequential(
                    tnn.Linear(obs_dim, hidden), tnn.Tanh(),
                    tnn.Linear(hidden, hidden),  tnn.Tanh(),
                )
                self.mean_head   = tnn.Linear(hidden, act_dim)
                self.log_std     = tnn.Parameter(torch.zeros(act_dim))
                self.value_head  = tnn.Linear(hidden, 1)

            def forward(self, obs):
                feat  = self.shared(obs)
                mean  = torch.tanh(self.mean_head(feat))  # clip to [-1,1]
                value = self.value_head(feat).squeeze(-1)
                return mean, self.log_std.exp(), value

            def get_action(self, obs):
                mean, std, value = self(obs)
                dist   = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_p  = dist.log_prob(action).sum(-1)
                return action, log_p, value

            def evaluate(self, obs, actions):
                mean, std, value = self(obs)
                dist  = torch.distributions.Normal(mean, std)
                log_p = dist.log_prob(actions).sum(-1)
                ent   = dist.entropy().sum(-1)
                return log_p, ent, value

        return GaussianPolicy(self.obs_dim, self.act_dim, hidden_dim)

    def _make_env(self, task, seed):
        return AntPinpad(task, **self.env_kwargs, seed=int(seed))

    def _augment_obs(self, env: AntPinpad, obs: np.ndarray) -> np.ndarray:
        direction = get_direction_augmentation(env)
        return np.concatenate([obs, direction], dtype=np.float32)

    def _shaped_reward(
        self,
        env: AntPinpad,
        reward: float,
        direction: np.ndarray,
    ) -> float:
        """Add intrinsic reward: dot(xy_velocity, direction_vector)."""
        vx = float(self.env_kwargs.get('_vx_override', env._data.qvel[0]))
        vy = float(self.env_kwargs.get('_vy_override', env._data.qvel[1]))
        vel = np.array([vx, vy])

        target_dir = CARDINAL_WORLD_VECS.T @ direction
        intrinsic  = float(np.dot(vel, target_dir))
        return reward + self.intrinsic_scale * intrinsic

    def train(self, verbose: bool = True) -> None:
        """Run PPO training."""
        import torch
        import torch.nn.functional as F

        device = self.device
        policy = self.policy

        total_steps = 0
        update_num  = 0

        # Pick a random task and make env
        task = self.tasks[int(self._rng.integers(len(self.tasks)))]
        env  = self._make_env(task, seed=self._rng.integers(1e9))
        obs_np = env.reset()
        obs_aug = self._augment_obs(env, obs_np)

        while total_steps < self.train_steps:
            # --- Collect rollout ---
            obs_buf    = np.zeros((self.rollout_steps, self.obs_dim), dtype=np.float32)
            act_buf    = np.zeros((self.rollout_steps, self.act_dim), dtype=np.float32)
            rew_buf    = np.zeros(self.rollout_steps, dtype=np.float32)
            done_buf   = np.zeros(self.rollout_steps, dtype=bool)
            logp_buf   = np.zeros(self.rollout_steps, dtype=np.float32)
            val_buf    = np.zeros(self.rollout_steps, dtype=np.float32)

            for t in range(self.rollout_steps):
                obs_t = torch.tensor(obs_aug, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, logp_t, val_t = policy.get_action(obs_t)

                action_np = action_t.squeeze(0).cpu().numpy()
                direction = get_current_direction(env)

                next_obs_np, rew, done, info = env.step(action_np)
                shaped_rew = self._shaped_reward(env, rew, direction)

                obs_buf[t]  = obs_aug
                act_buf[t]  = action_np
                rew_buf[t]  = shaped_rew
                done_buf[t] = done
                logp_buf[t] = logp_t.item()
                val_buf[t]  = val_t.item()

                if done:
                    task    = self.tasks[int(self._rng.integers(len(self.tasks)))]
                    env     = self._make_env(task, seed=int(self._rng.integers(1e9)))
                    obs_np  = env.reset()
                    obs_aug = self._augment_obs(env, obs_np)
                else:
                    obs_np  = next_obs_np
                    obs_aug = self._augment_obs(env, obs_np)

            total_steps += self.rollout_steps

            # Bootstrap value
            obs_t = torch.tensor(obs_aug, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, last_val = policy(obs_t)
            last_val = last_val.item() * (not done_buf[-1])

            # GAE
            adv_buf  = np.zeros_like(rew_buf)
            ret_buf  = np.zeros_like(rew_buf)
            gae      = 0.0
            for t in reversed(range(self.rollout_steps)):
                next_val = last_val if t == self.rollout_steps - 1 else val_buf[t + 1]
                delta    = rew_buf[t] + self.gamma * next_val * (1 - done_buf[t]) - val_buf[t]
                gae      = delta + self.gamma * self.gae_lambda * (1 - done_buf[t]) * gae
                adv_buf[t]  = gae
                ret_buf[t]  = gae + val_buf[t]

            # Normalise advantages
            adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

            # PPO update
            obs_t   = torch.tensor(obs_buf,  device=device)
            act_t   = torch.tensor(act_buf,  device=device)
            logp_t  = torch.tensor(logp_buf, device=device)
            adv_t   = torch.tensor(adv_buf,  device=device, dtype=torch.float32)
            ret_t   = torch.tensor(ret_buf,  device=device, dtype=torch.float32)

            idx = np.arange(self.rollout_steps)
            for _ in range(self.n_epochs):
                self._rng.shuffle(idx)
                for start in range(0, self.rollout_steps, self.batch_size):
                    batch = idx[start : start + self.batch_size]
                    new_logp, ent, val_pred = policy.evaluate(obs_t[batch], act_t[batch])
                    ratio    = (new_logp - logp_t[batch]).exp()
                    surr1    = ratio * adv_t[batch]
                    surr2    = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[batch]
                    pol_loss = -torch.min(surr1, surr2).mean()
                    val_loss = F.mse_loss(val_pred, ret_t[batch])
                    ent_loss = -ent.mean()
                    loss     = pol_loss + 0.5 * val_loss + self.entropy_coef * ent_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    self.optimizer.step()

            update_num += 1
            if verbose and update_num % 10 == 0:
                print(f"  [PPO] steps={total_steps:,}/{self.train_steps:,} "
                      f"loss={loss.item():.4f}")

        self._trained = True
        print(f"[PPO] Training complete. Total steps: {total_steps:,}")

    def save(self, path: str) -> None:
        import torch
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        import torch
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self._trained = True

    def get_action_mean(self, env: AntPinpad, obs: np.ndarray) -> np.ndarray:
        """Return mean action (no sampling) — used for clean dataset collection."""
        import torch
        obs_aug = self._augment_obs(env, obs)
        obs_t   = torch.tensor(obs_aug, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = self.policy(obs_t)
        return mean.squeeze(0).cpu().numpy()


def _torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Ant trajectory collection
# ---------------------------------------------------------------------------

def collect_ant_trajectory(
    task:          Tuple[int, ...],
    policy,                          # AntPPOExpert instance (already trained)
    use_mean:      bool  = True,     # paper: use mean for clean dataset
    grid_size:     int   = 4,
    n_objects:     int   = 4,
    n_walls:       int   = 1,
    max_steps:     int   = 500,
    cell_size:     float = 2.0,
    torso_z_min:   float = 0.2,
    torso_z_max:   float = 2.0,
    seed:          Optional[int] = None,
) -> Dict[str, Any]:
    """
    Roll out the trained PPO policy (mean actions) for one ant-pinpad episode.

    Returns dict with same structure as collect_gridworld_trajectory.
    """
    env_kwargs = dict(
        grid_size=grid_size, n_objects=n_objects, n_walls=n_walls,
        max_steps=max_steps, cell_size=cell_size,
        torso_z_min=torso_z_min, torso_z_max=torso_z_max,
    )
    env = AntPinpad(task, **env_kwargs, seed=seed)
    obs = env.reset()

    observations     = [obs.copy()]
    actions          = []
    rewards          = []
    dones            = []
    abstract_actions = []

    done = False
    info = {}
    while not done:
        if use_mean:
            action = policy.get_action_mean(env, obs)
        else:
            import torch
            obs_aug = policy._augment_obs(env, obs)
            obs_t   = torch.tensor(obs_aug, dtype=torch.float32,
                                   device=policy.device).unsqueeze(0)
            with torch.no_grad():
                act_t, _, _ = policy.policy.get_action(obs_t)
            action = act_t.squeeze(0).cpu().numpy()

        obs, reward, done, info = env.step(action)
        actions.append(action.copy())
        rewards.append(reward)
        dones.append(done)
        abstract_actions.append(info.get('abstract_action', env._goal_idx))
        observations.append(obs.copy())

    return dict(
        observations      = np.array(observations,     dtype=np.float32),
        actions           = np.array(actions,          dtype=np.float32),
        rewards           = np.array(rewards,          dtype=np.float32),
        dones             = np.array(dones,            dtype=bool),
        abstract_actions  = np.array(abstract_actions, dtype=np.int32),
        task              = task,
        success           = info.get('success', False),
    )


def collect_ant_dataset(
    tasks:               List[Tuple[int, ...]],
    policy,
    n_episodes_per_task: int   = 100,
    only_successful:     bool  = True,
    use_mean:            bool  = True,
    base_seed:           int   = 0,
    max_attempts_per_task: Optional[int] = None,
    **env_kwargs,
) -> List[Dict[str, Any]]:
    dataset = []
    for task_idx, task in enumerate(tasks):
        collected = 0
        attempt   = 0
        max_attempts = max_attempts_per_task or max(200, 50 * n_episodes_per_task)
        while collected < n_episodes_per_task:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Could only collect {collected}/{n_episodes_per_task} episodes "
                    f"for task {task} after {attempt} attempts."
                )
            seed = base_seed + task_idx * 10000 + attempt
            traj = collect_ant_trajectory(
                task=task, policy=policy, use_mean=use_mean,
                seed=seed, **env_kwargs
            )
            attempt += 1
            if only_successful and not traj['success']:
                continue
            dataset.append(traj)
            collected += 1
        print(f"  Task {task}: {collected} episodes "
              f"(success rate {collected/attempt:.1%})")
    return dataset
