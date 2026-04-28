import gymnasium as gym

gym.register(
    id="AntPinpad-v0",
    entry_point="envs.ant_pinpad_gym:AntPinpadGym",
    max_episode_steps=500,
)
