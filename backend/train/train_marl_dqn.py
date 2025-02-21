import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from env.supply_chain_env import SupplyChainEnv

# Load dataset
df = pd.read_csv("C:\\Users\\Dell\\Downloads\\construction-optimisation\\construction_marl_dataset.csv")

# Create and vectorize the environment
env = make_vec_env(lambda: SupplyChainEnv(df), n_envs=1)

# Train a Deep Q-Network (DQN) model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, batch_size=32)
model.learn(total_timesteps=10000)

# Save trained model
model.save("models/supply_chain_marl_dqn")

print("✅ Model training complete and saved!")
