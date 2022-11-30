from gym.envs.registration import register

register(
    id='grasp-v0',
    entry_point='gym_grasp.envs:GraspEnv',
    max_episode_steps=300,
)