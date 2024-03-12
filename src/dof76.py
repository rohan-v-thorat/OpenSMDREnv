""""
Authors of this code:
Rohan Thorat, Juhi Singh, Prof. Rajdip Nayek

Default system parameter values are taken from this article: https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9399%282004%29130%3A4%28437%29

"""

import gymnasium as gym
from scipy.linalg import expm
from gym import spaces
from gym.utils import seeding
import numpy as np