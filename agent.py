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

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.get_sample()
        next_q_values = self.q_target_network.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            actions_one_hot = tf.one_hot(actions, self.action_space)
            q_values_for_actions = tf.reduce_sum(q_values * actions_one_hot, axis=1)
            loss = self.loss(target_q_values, q_values_for_actions)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def train(self):
        for epoch in range(self.max_epochs):
            state = self.env.reset()
            total_reward = 0
            for step in range(self.max_steps):
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
                    q_values = self.q_network(state_tensor, training=False)
                    action = np.argmax(q_values.numpy())

                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                self.train_step()

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)
            print(f"Epoch: {epoch + 1}, Total Reward: {total_reward}")

    def save_model(self, path):
        self.q_network.save(path)

    def load_model(self, path):
        self.q_network = tf.keras.models.load_model(path)
        self.q_target_network.set_weights(self.q_network.get_weights())