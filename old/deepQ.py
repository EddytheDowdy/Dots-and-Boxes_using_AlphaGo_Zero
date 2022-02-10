import tensorflow as tf
from collections import deque
from old.Board import DabBoard
import numpy as np

env = DabBoard(4,4)
num_features = env.num_bordes
num_actions = env.num_bordes


class DQN(tf.keras.Model):

    def __init__(self, hidden_dim):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32)

    def __call__(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


main_nn = DQN(hidden_dim=128)
target_nn = DQN(hidden_dim=128)

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()


class ReplayBuffer(object):

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)

        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones


def select_epsilon_greedy_action(state, epsilon):
    result = tf.random.uniform((1,))
    is_eps = result < epsilon
    if is_eps:
        return env.action_space_sample(), is_eps
    else:
        return main_nn(state)[0], is_eps


@tf.function
def train_step(states, actions, rewards, next_states, dones):
    next_qs = target_nn(next_states)
    max_next_qs = tf.reduce_max(next_qs, axis=-1)
    target = rewards + (1. - dones) * discount * max_next_qs

    with tf.GradientTape() as tape:
        qs = main_nn(states)
        action_masks = tf.one_hot(actions, num_actions)
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
        loss = mse(target, masked_qs)

    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
    return loss


num_episodes = 1000
epsilon = 0.05
min_epsilon = 0.05
batch_size = 32
discount = 0.99
buffer = ReplayBuffer(100000)
cur_frame = 0

last_100_ep_rewards = []

for episode in range(num_episodes + 1):
    state = env.reset()
    mask = np.ones(num_actions)
    ep_reward, done = 0, False
    while not done:
        state_in = tf.expand_dims(state, axis=0)
        nn_output, is_epsilon = select_epsilon_greedy_action(state_in, epsilon)
        if is_epsilon:
            action = nn_output
        else:
            nn_masked = nn_output * mask
            action = tf.argmax(nn_masked).numpy()
        mask[action] = -np.inf
        _, reward, next_state, done, info = env.jugada(action, False)
        ep_reward += reward
        # Save to experience replay.
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        cur_frame += 1
        # Copy main_nn weights to target_nn.
        if cur_frame % 2000 == 0:
            target_nn.set_weights(main_nn.get_weights())

        # Paso de entrenamiento de la red neuronal principal.
        if len(buffer) >= batch_size:
            # extrae datos del buffer
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            loss = train_step(states, actions, rewards, next_states, dones)

    if epsilon > min_epsilon:
        epsilon -= 0.001

    if len(last_100_ep_rewards) == 100:
        last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(ep_reward)

    if episode % 50 == 0:
        print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
              f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
