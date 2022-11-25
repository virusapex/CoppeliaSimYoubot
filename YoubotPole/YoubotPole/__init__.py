from gym.envs.registration import register


register(
    id='YoubotPole-v0',
    entry_point='YoubotPole.envs:YoubotPoleEnv',
    max_episode_steps=10000,
)
