import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import gymnasium as gym
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from stable_baselines3 import SAC # if this doesnt work, uninstall reinstall torch
from stable_baselines3.common.env_util import make_vec_env

from walker2d_env import Walker2dEnv

np.set_printoptions(precision=3, suppress=True, linewidth=100)

env_name = "Walker2DBulletEnv-v0"
env = gym.make(env_name, render_mode='rgb_array')

model = SAC(
    "MlpPolicy",
    env,
    tensorboard_log="./sac_logs/"
)

model.learn(total_timesteps=1_000, progress_bar=True)
model.save("test_run")