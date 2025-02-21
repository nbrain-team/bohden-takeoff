import gym
import numpy as np
import pandas as pd
from gym import spaces

class SupplyChainEnv(gym.Env):
    def __init__(self, df):
        super(SupplyChainEnv, self).__init__()

        self.df = df
        self.num_suppliers = len(df)
        
        # Actions: Selecting a supplier (discrete choices)
        self.action_space = spaces.Discrete(self.num_suppliers)

        # Observations: Supplier attributes (cost, rating, delivery time)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Resets environment state."""
        self.current_supplier = np.random.randint(0, self.num_suppliers)
        return self._get_obs()

    def _get_obs(self):
        """Returns normalized supplier attributes."""
        supplier = self.df.iloc[self.current_supplier]
        return np.array([
            supplier["Cost per unit (INR)"] / 10000,  
            supplier["Rating"] / 5,  
            supplier["Average Delivery Time (days)"] / 30  
        ], dtype=np.float32)

    def step(self, action):
        """Takes an action (selecting a supplier) and returns reward."""
        selected_supplier = self.df.iloc[action]
        
        cost = selected_supplier["Cost per Unit (INR)"]
        rating = selected_supplier["Rating"]
        delivery_time = selected_supplier["Average Delivery Time (days)"]

        # Reward function: Higher rating, lower cost & delivery time
        reward = (rating * 10) - (cost / 1000) - (delivery_time / 5)

        done = True  # One-step decision process
        return self._get_obs(), reward, done, {}

