""""
Authors of this code:
Rohan Thorat, Juhi Singh, Prof. Rajdip Nayek

Default system parameter values are taken from this article: https://ascelibrary.org/doi/10.1061/%28ASCE%29EM.1943-7889.0001226

"""

import gymnasium as gym
from scipy.linalg import expm
from gym import spaces
from gym.utils import seeding
import numpy as np