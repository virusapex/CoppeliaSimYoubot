from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback

from YoubotPole.YoubotPoleEnv_zmq import YoubotPoleEnv


if __name__ == "__main__":
    # ---------------- Create environment
    # env = YoubotPoleEnv()
    env = SubprocVecEnv(
        [lambda: Monitor(YoubotPoleEnv(23000)), lambda: Monitor(YoubotPoleEnv(23002))])
    # check_env(env)

    # ---------------- Callback functions
    # env = Monitor(env)
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 250000,
        "env_name": "YoubotPoleEnv",}

    run = wandb.init(
        project="youbot_pole",
        config=config,
        sync_tensorboard=True,
        save_code=True)

    # ---------------- Model
    # model = A2C(
    #     config["policy_type"],
    #     env,
    #     verbose=1,
    #     tensorboard_log=f"runs/{run.id}")
    model = PPO(
        config["policy_type"],
        env,
        use_sde=False,
        verbose=1,
        tensorboard_log=f"runs/{run.id}")
    # model = RecurrentPPO(
    #     "MlpLstmPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log=f"runs/{run.id}")

    # # ---------------- Learning
    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(
                    model_save_freq=10000,
                    model_save_path=f"models/{run.id}",
                    verbose=2))
                
    print('Finished')
    del model

    # model = A2C.load(f"models/{run.id}/model", env=env)
    env = YoubotPoleEnv(23000)
    model = PPO.load(f"models/{run.id}/model", env=env)
    # model = RecurrentPPO.load(f"models/{run.id}/model", env=env)

    # ---------------- Prediction
    print('Prediction')

    for _ in range(10):
        observation, done = env.reset(), False
        episode_reward = 0.0

        num_envs = 1
        # Episode start signals are used to reset the lstm states
        # lstm_states = None
        # episode_starts = np.ones((num_envs,), dtype=bool)
        # states = model.initial_state
        # dones = np.zeros(states.shape[0])

        while not done:
            # action, lstm_states = model.predict(
            #     observation,
            #     state=lstm_states,
            #     episode_start=episode_starts,
            #     deterministic=True)
            action, _state = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
        
        print([episode_reward])

    env.close()
