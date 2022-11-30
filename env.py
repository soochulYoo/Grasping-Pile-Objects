import gym_grasp
from mjrl.utils.gym_env import GymEnv

def make_env(env_name):
    env = GymEnv(env_name)
    return env