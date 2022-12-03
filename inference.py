import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from YoubotPole.envs.YoubotPoleEnv import YoubotPoleEnv


parser = argparse.ArgumentParser()
parser.add_argument("--algo", help="RL Algorithm", 
                    default="ppo", type=str, required=False)
parser.add_argument("-i", "--trained-agent", help="Path to a trained agent",
                    default="best_model", type=str)
parser.add_argument("--norm", type=str,
                    default="vecnormalize.pkl", help="Path to a VecNormalize statistics")
parser.add_argument("--env", type=str,
                    default="YoubotPole-v0", help="Environment ID")

args = parser.parse_args()


if args.env == "YoubotPole-v0":
    env = DummyVecEnv([lambda: YoubotPoleEnv(23000)])
    env = VecNormalize.load(args.norm, env)
    env.training = False
    env.norm_reward = False

if args.algo == "ppo":
    model = PPO.load(args.trained_agent, env=env)
elif args.algo == "ppo_lstm":
    model = RecurrentPPO.load(args.trained_agent, env=env)

# ---------------- Prediction
print('Prediction')

for _ in range(10):
    observation, done = env.reset(), False
    episode_reward = 0.0

    num_envs = 1
    # Episode start signals are used to reset the lstm states
    if args.algo == "ppo_lstm":
        lstm_states = None
        episode_starts = np.ones((num_envs,), dtype=bool)


    while not done:
        if args.algo == "ppo_lstm":
            action, lstm_states = model.predict(
                observation,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True)
        else:
            action, _state = model.predict(observation, deterministic=True)

        observation, reward, done, info = env.step(action)
        episode_reward += reward
        episode_starts = done
    
    print([episode_reward])

env.close()
