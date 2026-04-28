"""
Task definitions for gridworld-pinpad and ant-pinpad environments.

Each task is a tuple of color IDs the agent must visit in order.
Color IDs are integers starting from 0.

From the paper (Appendix A):
  Gridworld: G=7, O=8 colors (0-7), W=4 walls, T=100
    Abstract subgoal pairs: (0,1), (2,3), (4,5), (6,7)
    Post-training task: 0-1-2-3-4-5-6-7-0-1-2-3

  Ant: G=4, O=4 colors (0-3), W=1 wall, T=500
    Post-training task: 0-1-2-3
"""

# ---------------------------------------------------------------------------
# Gridworld-pinpad tasks (Table A1 in paper)
# ---------------------------------------------------------------------------

GRIDWORLD_PRETRAINING_TASKS = [
    (0, 1, 4, 5, 0, 1),
    (0, 1, 4, 5, 2, 3),
    (0, 1, 6, 7, 2, 3),
    (2, 3, 0, 1, 4, 5),
    (2, 3, 6, 7, 2, 3),
    (2, 3, 6, 7, 4, 5),
    (4, 5, 0, 1, 4, 5),
    (4, 5, 0, 1, 6, 7),
    (4, 5, 2, 3, 6, 7),
    (6, 7, 2, 3, 0, 1),
    (6, 7, 2, 3, 6, 7),
    (6, 7, 4, 5, 0, 1),
    (0, 1, 6, 7, 4, 5),
    (2, 3, 0, 1, 6, 7),
    (4, 5, 2, 3, 0, 1),
    (6, 7, 4, 5, 2, 3),
]

GRIDWORLD_POSTTRAINING_TASK = (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3)

# Abstract subgoal pairs: visiting both colors in a pair = one abstract action
GRIDWORLD_SUBGOAL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

# ---------------------------------------------------------------------------
# Ant-pinpad tasks (Table A2 in paper)
# ---------------------------------------------------------------------------

ANT_PRETRAINING_TASKS = [
    (0, 3, 2),
    (1, 0, 3),
    (2, 1, 0),
    (3, 2, 1),
    (0, 2, 0),
    (0, 2, 1),
    (0, 3, 1),
    (1, 0, 2),
    (1, 3, 1),
    (1, 3, 2),
    (2, 0, 2),
    (2, 0, 3),
    (2, 1, 3),
    (3, 1, 0),
    (3, 1, 3),
    (3, 2, 0),
]

ANT_POSTTRAINING_TASK = (0, 1, 2, 3)

# ---------------------------------------------------------------------------
# Environment hyperparameters (Appendix A)
# ---------------------------------------------------------------------------

GRIDWORLD_CONFIG = dict(
    grid_size=7,   # G: grid is G×G
    n_objects=8,   # O: number of colored cells
    n_walls=4,     # W: number of wall cells
    max_steps=100, # T
)

ANT_CONFIG = dict(
    grid_size=4,   # G
    n_objects=4,   # O
    n_walls=1,     # W
    max_steps=500, # T
    torso_z_min=0.2,
    torso_z_max=1.0,
    cell_size=2.0,
)
