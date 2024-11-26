import gym
from gym import spaces
import numpy as np
import random

class HybridTradingEnv(gym.Env):
    def __init__(self, real_data, simulated_datasets, initial_balance=10000):
        super(HybridTradingEnv, self).__init__()
        self.simulated_datasets = simulated_datasets
        self.real_data = real_data
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.state_space = len(simulated_datasets[0].columns) + 2  # Datos + balance + posición
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(self.state_space,), dtype=np.float32)

        self.data = None
        self.current_step = 0
        self.balance = 0
        self.position = 0
        self.max_assets = 0

    def reset(self):
        self.data = random.choice(self.simulated_datasets + self.real_data)
        self.n_steps = len(self.data)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.max_assets = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        current_data = self.data.iloc[self.current_step].values
        return np.concatenate((current_data, [self.balance, self.position]))

    def step(self, action):
        current_price = self.data.iloc[self.current_step]["Close"]
        previous_assets = self.balance + self.position * current_price

        if action == 1:  # Buy
            if self.balance >= current_price:
                self.position += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.position > 0:
                self.position -= 1
                self.balance += current_price

        total_assets = self.balance + self.position * current_price
        self.max_assets = max(self.max_assets, total_assets)
        drawdown = (self.max_assets - total_assets) / self.max_assets

        # Recompensa ajustada
        reward = (total_assets - previous_assets) / previous_assets - drawdown - (0.01 if action == 0 else 0)

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        return self._get_observation(), reward, done, {}

    def render(self, mode="human"):
        current_price = self.data.iloc[self.current_step]["Close"]
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
              f"Position: {self.position}, Current Price: {current_price:.2f}")
