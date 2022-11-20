from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, A2C

import wandb
from wandb.integration.sb3 import WandbCallback

from YoubotPole.YoubotPoleEnv import YoubotPoleEnv


# ---------------- Create environment
env = YoubotPoleEnv(action_type='continuous')

check_env(env)

# ---------------- Callback functions
env = Monitor(env)

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 250000,
    "env_name": "YoubotPoleEnv",
}

run = wandb.init(
    project="youbot_pole",
    config=config,
    sync_tensorboard=True,
    save_code=True
)

# ---------------- Model
# create a ppo model
print("ppo model creation")
# model = PPO(
#     config["policy_type"],
#     env,
#     verbose=1,
#     use_sde=False,
#     tensorboard_log=f"runs/{run.id}"
# )

model = A2C(
    config["policy_type"],
    env,
    verbose=1,
    use_sde=False,
    tensorboard_log=f"runs/{run.id}"
)

# ---------------- Learning
print('Learning the model')
model.learn(total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
                )
            )
            
print('Finished')
del model

model = PPO.load(f"models/{run.id}/model", env=env)

# ---------------- Prediction
print('Prediction')

for _ in range(10):
    observation, done = env.reset(), False
    episode_reward = 0.0

    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
    
    print([episode_reward, env.counts])

env.close()
