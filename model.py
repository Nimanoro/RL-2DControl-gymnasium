import numpy as np
import random
from acrobat import start_env
import matplotlib.pyplot as plt
def discretize_state(state, bins):
    state_bins = [
        np.linspace(-1.0, 1.0, bins),  # cos(theta1)
        np.linspace(-1.0, 1.0, bins),  # sin(theta1)
        np.linspace(-1.0, 1.0, bins),  # cos(theta2)
        np.linspace(-1.0, 1.0, bins),  # sin(theta2)
        np.linspace(-12.0, 12.0, bins),  # angular velocity 1
        np.linspace(-28.0, 28.0, bins),  # angular velocity 2
    ]
    discretized = tuple(
        np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state))
    )
    return discretized

def init_q_table(state_bins, action_space):
    Qtable = np.zeros(state_bins + [action_space])
    return Qtable

def epsilon_greedy_policy(Qtable, state, epsilon):
    rand = random.uniform(0, 1)
    if rand > epsilon:
        action = np.argmax(Qtable[state])
    else:
        action = env.action_space.sample()
    return action

def greedy_policy(Qtable, state):
    return np.argmax(Qtable[state])

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    learning_rate = 0.5  # Adjusted for smoother updates
    gamma = 0.99  # Focus on long-term rewards
    rewards_log = []  # To track rewards per episode
    epsilon_log = []  # To track epsilon decay

    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        epsilon_log.append(epsilon)

        obs, _ = env.reset()
        state = discretize_state(obs, 15)  # Updated to 15 bins
        total_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, done, _ = env.step(action)
            new_state = discretize_state(new_state, 15)

            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            state = new_state
            total_reward += reward

            if terminated or done:
                break

        rewards_log.append(total_reward)

        # Print progress during training
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_training_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

    return Qtable, rewards_log, epsilon_log

def test_model(Qtable, env, n_eval):
    rewards = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        state = discretize_state(obs, 15)
        total_reward = 0
        done = False

        while not done:
            action = greedy_policy(Qtable, state)
            new_state, reward, terminated, done, _ = env.step(action)
            state = discretize_state(new_state, 15)
            total_reward += reward

        rewards.append(total_reward)

    return rewards

# Initialize the environment and parameters
env = start_env()
action_space = env.action_space.n
state_bins = [15] * 6  # Updated binning
Qtable_start = init_q_table(state_bins, action_space)

# Training parameters
n_training_episodes = 500  # Increased episodes
max_steps = 200
n_eval = 100
decay_rate = 0.009  # Adjusted decay rate
min_eps = 0.05
max_eps = 1.0

# Train the agent
Qtable_acrobat, training_rewards, epsilon_log = train(
    n_training_episodes, min_eps, max_eps, decay_rate, env, max_steps, Qtable_start
)

# Evaluate the model
rewards = test_model(Qtable_acrobat, env, n_eval)

# Logging and visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(training_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Rewards")

plt.subplot(1, 2, 2)
plt.plot(epsilon_log)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay")

plt.tight_layout()
plt.show()

# Display evaluation results
print("Average Test Reward:", np.mean(rewards))
print("Standard Deviation of Test Rewards:", np.std(rewards))
