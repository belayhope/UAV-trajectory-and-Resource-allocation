# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#withoutGMM-MADDPG
#import library
from collections import deque # Import deque from collections module
from IPython import get_ipython
from IPython.display import display
import os
import pandas as pd
import tensorflow as tf
import gym
print("TensorFlow version:", tf.__version__)
print("Gym version:", gym.__version__)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
import collections
import numpy as np
from scipy.stats import multivariate_normal  # Import for Gaussian PDF
from sklearn.cluster import KMeans  # Import for KMeans initialization
import csv  # Import the csv module
from IPython import get_ipython
from IPython.display import display
from IPython.display import display
from IPython import get_ipython
import scipy.interpolate as si
import random
#network parameter
EPISODES =1000
STEPS = 200
GAMMA = 0.98 # Discount factor
TAU = 0.001  # Target update rate
BUFFER_SIZE = 500000
BATCH_SIZE = 256
LR = 0.0001  # Learning rate
# Exploration strategy: Exponential decay for epsilon-greedy
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
thrashold_R=2.85 #Mbps
# Ornstein-Uhlenbeck Noise for Exploration
class OUActionNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = np.zeros_like(self.mean)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x
# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, size=BUFFER_SIZE):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)

def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same state.
    The distance is the difference between the actions taken by the policies.
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np
                # DDPG Agent Class
class DDPGAgent:
     def __init__(self, state_dim, action_dim, tau=0.001, entropy_weight=0.01, clip_norm=1.0):
        self.state_dim = state_dim  # Now uses the updated state_dim from UAVEnvironment
        self.action_dim = action_dim
        self.tau = tau
        self.entropy_weight = entropy_weight
        self.clip_norm = clip_norm
        self.buffer = ReplayBuffer()  # Assuming ReplayBuffer is defined elsewhere
        self.actor = self.build_actor()
        self.target_actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_critic = self.build_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        if tf:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        else:
            self.optimizer = None  # Or a suitable alternative if available
        self.noise = OUActionNoise(mean=np.zeros(self.action_dim), std_dev=0.2 * np.ones(self.action_dim))

     def build_actor(self):
            model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='tanh')
        ])
            print("Actor model created successfully.")
            return model

     def build_critic(self):
        if not tf:
            return None
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim + self.action_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model

     def select_action(self, state):
        if self.actor is None:
            return np.random.uniform(-1, 1, self.action_dim)
        state = np.expand_dims(state, axis=0)
        action = self.actor(state).numpy()[0] + self.noise()
        return np.clip(action, -1, 1)


     def actor_loss(self, states, actions):
        if not tf:
            return None
        with tf.GradientTape() as tape:
            q_values = self.critic(tf.concat([states, actions], axis=1))
            # Entropy Regularization
            entropy = -tf.reduce_sum(tf.nn.softmax(actions) * tf.math.log(tf.nn.softmax(actions) + 1e-8), axis=-1)
            loss = -tf.reduce_mean(q_values) + self.entropy_weight * tf.reduce_mean(entropy)
            # Trust Region Policy Updates (Clipping)
            gradients = tape.gradient(loss, self.actor.trainable_variables)
            clipped_gradients = [tf.clip_by_norm(grad, clip_norm=self.clip_norm) for grad in gradients]
            self.optimizer.apply_gradients(zip(clipped_gradients, self.actor.trainable_variables))
        return loss, tape  # Return loss and tape if needed

     def update_target_networks(self):
        if not tf:
            return
        # Soft Target Update
        new_weights = []
        target_weights = self.target_actor.get_weights()
        for i, weight in enumerate(self.actor.get_weights()):
            new_weights.append(weight * self.tau + target_weights[i] * (1 - self.tau))
        self.target_actor.set_weights(new_weights)

        new_weights = []
        target_weights = self.target_critic.get_weights()
        for i, weight in enumerate(self.critic.get_weights()):
            new_weights.append(weight * self.tau + target_weights[i] * (1 - self.tau))
        self.target_critic.set_weights(new_weights)
        # Federated Learning Server
import threading
import time
class FederatedServer:
    def __init__(self, agents, shared_layers):
        self.agents = agents
        self.shared_layers = shared_layers
        self.weight_buffer = {}
        self.lock = threading.Lock()

    def receive_weights(self, agent_id, weights):
        with self.lock:
            self.weight_buffer[agent_id] = weights
            self.aggregate_weights()

    def aggregate_weights(self):
        if not self.weight_buffer:
            return

        agent_weights = [self.weight_buffer[agent_id]
                         for agent_id in self.weight_buffer
                         if self.agents[agent_id].actor is not None]

        if not agent_weights:
            return

        new_weights = [None] * len(agent_weights[0])
        for layer_index in self.shared_layers:
            layer_weights = [agent_weights[agent_index][layer_index]
                             for agent_index in range(len(agent_weights))]
            avg_layer_weights = np.mean(layer_weights, axis=0)
            new_weights[layer_index] = avg_layer_weights

        for agent in self.agents:
            if agent.actor is not None:
                current_weights = agent.actor.get_weights()
                for layer_index in self.shared_layers:
                    current_weights[layer_index] = new_weights[layer_index]
                agent.actor.set_weights(current_weights)
                # Recreate the optimizer after updating weights
                agent.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)  # Recreate optimizer
                # Training Loop
env = withoutGMM_Env()
agents = [DDPGAgent(env.state_dim, env.action_dim) for _ in range(NUM_UAVS)]
shared_layers = [0, 1]

# Create the FederatedServer instance with both arguments
server = FederatedServer(agents, shared_layers)
rewards = [] # Initialize rewards as a list
normalized_rewards = []
bandwidth_allocations = []
avg_data_rates = []
coverages = []
fairnesses = []
all_miotd_data_rates = []
coverage_data = []  # List to store coverage values for saving to CSV
user_load_fairnesses = []
bandwidth_fairnesses = []
moving_avg_rewards = []  # List to store moving average of rewards
window_size = 10  # Size of the moving average window
uav_positions_history = []  # Store UAV positions for each episode
fairness_data = []
miotd_load_fairnesses_per_uav = []
all_uav_miotd_counts = []
moving_avg_rewards = []
avg_data_rates = []
communication_interval = 10  # Initial communication interval
performance_threshold = 0.9  # Threshold for performance improvement
all_episodes_miotd_counts = []  # Store counts for all episodes
miotd_counts_per_uav_per_episode = []  # Initialize before the loop

#Training each episode
for episode in range(EPISODES):
    uav_positions_episode = []  # Store UAV positions for this episode
    state = env.reset()
    episode_reward = 0
    episode_miotd_counts = []
    actions = []
    for agent in agents:
        if np.random.rand() <= epsilon:
            action = np.random.uniform(-1, 1, env.action_dim)  # Random action
        else:
            action = agent.select_action(state)  # Agent's action
        actions.extend(action)  # Reset for each episode
    for step in range(STEPS):
        # The following line was not indented correctly:
        actions = [agent.select_action(state) for agent in agents]
        actions = np.concatenate(actions)  # Concatenate actions from all agents
        next_state, reward, _, info = env.step(actions)
        # Access miotd_load_fairness_per_uav from the info dictionary
        episode_reward += reward
        for agent in agents:
            agent.buffer.add((state, actions, reward, next_state))
        state = next_state

    # Perform Experience Replay
    if agents[0].buffer.size() > BATCH_SIZE:
        for agent in agents:
            minibatch = agent.buffer.sample(BATCH_SIZE)
            states, actions, rewards_batch, next_states = zip(*minibatch) # Change rewards to rewards_batch
            states = np.array(states, dtype=object)
            actions = np.array(actions)
            # rewards = []  # Remove this line - we don't want to reset rewards here
            next_states = np.array(next_states)


    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    # Get connected MIoTD indices
    connected_miotd_indices = np.sum(env.data_rates_mbps >= thrashold_R)
    uav_positions_history.append(uav_positions_episode)
    miotd_data_rates_episode = env.get_miotd_data_rates()
    all_miotd_data_rates.append(miotd_data_rates_episode)
    avg_data_rates.append(np.mean(env.data_rates_mbps))
    miotd_load_fairness_per_uav = info['miotd_load_fairness_per_uav']
    miotd_load_fairnesses_per_uav.append(miotd_load_fairness_per_uav)
    bandwidth_fairness = compute_bandwidth_fairness(env.bandwidth_allocation)
    uav_positions_episode.append(env.uav_positions.copy())  # Store a copy of UAV positions
    fairness_data.append([episode + 1, miotd_load_fairness_per_uav, bandwidth_fairness])
    user_load_fairnesses.append(miotd_load_fairness_per_uav) #Append the values here since now they are accessible in this scope.
    bandwidth_fairnesses.append(bandwidth_fairness)
    # Fix: Access data_rates_mbps from the env object
    coverage = connected_miotd_indices / NUM_MIOTD  # Calculate coverage here using data_rates_mbps from env
    coverages.append(coverage)
    coverage_data.append([episode + 1, coverage])  # Append coverage data for saving

    for episode_positions in uav_positions_history:
        # Get the UAV positions for the last step of the episode
        final_uav_positions = episode_positions[-1]
        # Calculate distances and data rates for the final UAV positions
        distances = np.linalg.norm(final_uav_positions[:, np.newaxis, :] - env.miotd_positions, axis=2)
        path_loss = CHANNEL_GAIN_REF / (distances ** 2 + 1e-9)
        signal_power = path_loss * (BANDWIDTH / 50)
        snr = signal_power / NOISE_POWER
        data_rates = (BANDWIDTH / 50) * np.log2(1 + snr)
        data_rates_mbps = data_rates / 1e6  # Convert to Mbps

        # Count MIoTDs connected to each UAV based on data rates and threshold
        miotd_counts_per_uav = [
            np.sum((np.argmax(data_rates_mbps, axis=0) == uav_index) & (data_rates_mbps[uav_index] >= thrashold_R))
            for uav_index in range(NUM_UAVS)
        ]
    # Append the counts for this episode to the overall list
    episode_miotd_counts.append(miotd_counts_per_uav)  # Append counts for this episode
    all_episodes_miotd_counts.extend(episode_miotd_counts)  # Add this episode's counts to the overall list

    # Modify this line to append to a list:
    miotd_counts_per_uav_per_episode.append(miotd_counts_per_uav)

    # Convert to NumPy array *after* appending:
    miotd_counts_per_uav_per_episode_np = np.array(miotd_counts_per_uav_per_episode)

    # Use the NumPy array for calculations:
    miotd_counts_current_episode = miotd_counts_per_uav_per_episode_np[-1] if miotd_counts_per_uav_per_episode_np.size > 0 else np.zeros(NUM_UAVS)


    #rewards = np.array(rewards) # Remove this line, keep rewards as a list
    rewards.append(episode_reward)
    bandwidth_allocations.append(env.bandwidth_allocation.copy())  # Store bandwidth allocation # Store bandwidth allocation

   # Calculate running mean and standard deviation of rewards
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    # Normalize the current episode reward
    normalized_reward = (episode_reward - mean_reward) / (std_reward + 1e-8)  # Avoid division by zero
    normalized_rewards.append(normalized_reward)  # Append the normalized reward to the list

    # In the training loop
    if episode % communication_interval == 0:
        server.aggregate_weights()

    # Print reward after each episode
    print(f"Episode {episode + 1}: Reward = {episode_reward}")

    # Check for performance improvement
    if np.mean(rewards[-communication_interval:]) > performance_threshold:
        communication_interval *= 2  # Increase interval if performance is good
    else:
        communication_interval = max(1, communication_interval // 2)  # Decrease if performance is poorserver.aggregate_weights()

    # Calculate and store moving average reward
    if episode >= window_size:
        moving_avg = np.mean(rewards[-window_size:])
        moving_avg_rewards.append(moving_avg)
    else:
        moving_avg_rewards.append(np.mean(rewards))


#save the result in csv



     # Save connected MIoTDs to CSV

    with open('connected_miotds.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Convert connected_miotd_indices to a list before concatenation
        writer.writerow([episode + 1] + [connected_miotd_indices])
    # Access coverage, fairness calculated in the step method

    df = pd.DataFrame(all_miotd_data_rates, columns=[f"MIoTD_{i}" for i in range(NUM_MIOTD)])
    df.to_csv("miotd_data_rates.csv", index=False)
    with open('miotd_counts_all_uavs.csv', 'w', newline='') as csvfile:
         writer = csv.writer(csvfile)
    # Write the header row
         writer.writerow(['Episode'] + [f'UAV_{i+1}' for i in range(NUM_UAVS)])
    # Write the data rows
         for episode_num, counts in enumerate(all_episodes_miotd_counts):  # Enumerate to get episode numbers
              writer.writerow([episode_num + 1] + counts)  # Add 1 to episode_num for 1-based indexing

#save reward
reward_data = {
    "Episode": range(1, EPISODES + 1),  # Episode numbers
    "Reward": rewards,
    "Normalized_Reward": normalized_rewards
}
reward_df = pd.DataFrame(reward_data)

# Save the DataFrame to a CSV file
reward_df.to_csv("reward_data.csv", index=False)  # index=False prevents saving the index column

moving_avg_data = []
for i, reward in enumerate(moving_avg_rewards):
    moving_avg_data.append([i + 1, reward])  # Episode number starts from 1
moving_avg_df = pd.DataFrame(moving_avg_data, columns=["Episode", "Moving_Avg_Reward"])
moving_avg_df.to_csv("moving_avg_rewards.csv", index=False)

#save data rate
data_rate_data = []
for i, data_rate in enumerate(avg_data_rates):
    data_rate_data.append([i + 1, data_rate])  # Episode number starts from 1
data_rate_df = pd.DataFrame(data_rate_data, columns=["Episode", "Avg_Data_Rate"])
data_rate_df.to_csv("avg_data_rates.csv", index=False)

#save fairness
fairness_df = pd.DataFrame(fairness_data, columns=["Episode", "User_Load_Fairness", "Bandwidth_Fairness"])
fairness_df.to_csv("fairness_data.csv", index=False)

#save coverage
coverage_df = pd.DataFrame(coverage_data, columns=["Episode", "Coverage"])
coverage_df.to_csv("coverage_data.csv", index=False)

#plot the result
#plot UAV trajectory
uav_colors = ['blue', 'black', 'green']  # Add more colors if you have more UAVs
for uav_index in range(NUM_UAVS):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Extract UAV positions for the current UAV
    x = []
    y = []
    z = []
    for episode_positions in uav_positions_history:
        x.extend([pos[uav_index, 0] for pos in episode_positions])
        y.extend([pos[uav_index, 1] for pos in episode_positions])
        z.extend([pos[uav_index, 2] for pos in episode_positions])

    # Plot UAV trajectory
    ax.plot(x, y, z, label=f"UAV {uav_index + 1}", linestyle='solid', marker='D', markersize=2, color=uav_colors[uav_index]) # Use color from uav_colors


    # Get connected MIoTDs for the final UAV position in the last episode
    final_uav_positions = uav_positions_history[-1][-1]  # UAV positions at the end of the last episode
    distances = np.linalg.norm(final_uav_positions[:, np.newaxis, :] - env.miotd_positions, axis=2)
    path_loss = CHANNEL_GAIN_REF / (distances ** 2 + 1e-9)
    signal_power = path_loss * (BANDWIDTH / 50)
    snr = signal_power / NOISE_POWER
    data_rates = (BANDWIDTH / 50) * np.log2(1 + snr)
    data_rates_mbps = data_rates / 1e6

    connected_miotd_indices = np.where(np.argmax(data_rates_mbps, axis=0) == uav_index)[0]

    # Plot connected MIoTDs in red
    ax.scatter(env.miotd_positions[connected_miotd_indices, 0],
               env.miotd_positions[connected_miotd_indices, 1],
               env.miotd_positions[connected_miotd_indices, 2],
               c='red', marker='o', label='Connected MIoTDs')

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title(f"UAV {uav_index + 1} Trajectory with Connected MIoTDs")
    ax.legend()
    plt.show()



# Plotting UAV Trajectories for all episodes in one plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Plot MIoTD locations
ax.scatter(env.miotd_positions[:, 0], env.miotd_positions[:, 1], env.miotd_positions[:, 2],
           c='red', marker='o', label='MIoTDs')

for uav_index in range(NUM_UAVS):
    x = []
    y = []
    z = []
    for episode_positions in uav_positions_history:
        x.extend([pos[uav_index, 0] for pos in episode_positions])
        y.extend([pos[uav_index, 1] for pos in episode_positions])
        z.extend([pos[uav_index, 2] for pos in episode_positions])

    # Plot with a different color for each episode
    ax.plot(x, y, z, label=f"UAV {uav_index + 1}",linestyle='-', marker='D',markersize=2)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("UAV Trajectories with MIoTD Locations")
ax.legend()  # Show legend to distinguish episodes and MIoTDs
plt.show()

uav_colors = ['blue', 'black', 'green']  # Colors for each UAV

plt.figure(figsize=(10, 10))  # Adjust figure size as needed

# Plot MIoTD locations
plt.scatter(env.miotd_positions[:, 0], env.miotd_positions[:, 1],
            c='red', marker='o', label='MIoTDs')

for uav_index in range(NUM_UAVS):
    x = []
    y = []
    for episode_positions in uav_positions_history:
        # Extract x and y coordinates for the current UAV
        x.extend([pos[uav_index, 0] for pos in episode_positions])
        y.extend([pos[uav_index, 1] for pos in episode_positions])

    # Plot the trajectory for the current UAV
    plt.plot(x, y, label=f"UAV {uav_index + 1}",
             linestyle='-', marker='D', markersize=2, color=uav_colors[uav_index])

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("UAV Trajectories in 2D Space with MIoTD Locations")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
for miotd_index in range(NUM_MIOTD):
    bandwidth_data = [bandwidths[:, miotd_index].sum() for bandwidths in bandwidth_allocations]  # Extract bandwidth for the MIoT
    plt.plot(range(1, len(bandwidth_data) + 1), bandwidth_data, label=f"MIoT {miotd_index + 1}")

plt.xlabel("Episode")
plt.ylabel("Allocated Bandwidth (Hz)")
plt.title("Allocated Bandwidth of Each MIoT vs Episode")
plt.legend()
plt.grid(True)
plt.show()

# Plot the results
plt.figure(figsize=(6, 5))
for uav_index in range(NUM_UAVS):
    # Modified plotting logic to use all_episodes_miotd_counts
    plt.plot(range(1, len(all_episodes_miotd_counts) + 1),
             [all_episodes_miotd_counts[episode - 1][uav_index] for episode in range(1, len(all_episodes_miotd_counts) + 1)],
             label=f"UAV {uav_index + 1}", marker='o')

plt.xlabel("Episode")
plt.ylabel("Number of MIoTDs Connected")
#plt.title("Number of MIoTDs Connected to Each UAV over Episodes")
plt.legend()
plt.grid(True)
plt.show()


# Plot Reward vs Episode
plt.figure(figsize=(10, 5))
plt.plot(range(1, episode + 2), rewards, label="Reward per Episode", color='red')
#plt.plot(range(1, episode + 2), normalized_rewards, label="Normalized Reward", color='r')  # Add normalized reward plot
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward vs Episode")
plt.legend()
plt.grid(True)
plt.show()

# Plot Avg Data Rate per Episode
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(avg_data_rates) + 1), avg_data_rates, label="Avg Data Rate per Episode", color='g', alpha=0.6)
plt.xlabel("Episode") # Change: Use xlabel to label the x-axis
plt.ylabel("Avg Data Rate(Mbps)")  # Corrected y-axis label
plt.title("Reward vs Episode (Training Performance)")
plt.legend()
plt.grid()
plt.show()


# Plot Coverage vs Episode
plt.figure(figsize=(10, 5)) # Change: Correct 'pplt' to 'plt'
plt.plot(range(1, len(coverages) + 1), coverages, label="Coverage per Episode", color='r',alpha=0.6)
plt.xlabel("Episode")
plt.ylabel("Coverage")
plt.title("Coverage vs Episode")
plt.legend()
plt.grid()


# Plotting MIoTD Load Fairness per UAV
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(miotd_load_fairnesses_per_uav) + 1), miotd_load_fairnesses_per_uav, label="MIoTD Load Fairness per UAV", color='orange')
plt.xlabel("Step")
plt.ylabel("Fairness Index")
plt.title("MIoTD Load Fairness per UAV over Steps")
plt.legend()
plt.grid()
plt.show()

# Plot Moving Average Reward vs Episode
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(moving_avg_rewards) + 1), moving_avg_rewards, label="GMM-FL-MADDPG", color='green')
plt.xlabel("Episode")
plt.ylabel("Reward")
#plt.title("Commulative Reward vs Episode")
plt.legend()
plt.grid(True)
plt.show()




