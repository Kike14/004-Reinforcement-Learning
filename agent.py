import tensorflow as tf
import numpy as np
from collections import deque

class Agent:
    def _init_(self, env, max_epochs=10, max_steps=500, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, batch_size=64,
                 learning_rate=0.0001, history_len=32000):
        self.env = env
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.history_len = history_len

        self.replay_buffer = deque(maxlen=self.history_len)
        self.action_space = env.action_space.n
        self.observation_space = env.observation_space.shape[0]

        self.q_network = self.init_q_network()
        self.q_target_network = self.init_q_network()
        self.q_target_network.set_weights(self.q_network.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = tf.keras.losses.Huber()

    def init_q_network(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.observation_space,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_space, activation='linear')
        ])
        return model