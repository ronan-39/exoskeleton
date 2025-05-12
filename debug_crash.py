import os
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["MUJOCO_GL"] = "osmesa"

import mujoco
import gymnasium as gym
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from stable_baselines3 import SAC # if this doesnt work, uninstall reinstall torch
from stable_baselines3.common.env_util import make_vec_env

from walker2d_env import Walker2dEnv

np.set_printoptions(precision=3, suppress=True, linewidth=100)

gym.register(
    id="gymnasium_env/Walker2d-v0",
    entry_point=Walker2dEnv,
    kwargs={'xml_file': './assets/walker2d.xml'}
)

env = gym.make('gymnasium_env/Walker2d-v0', render_mode=None)
# env = make_vec_env('gymnasium_env/Walker2d-v0', n_envs=1)

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_logs/")
model.learn(total_timesteps=1_000)