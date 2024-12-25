import numpy as np
import random
import os
import multiprocessing
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

def softmax(q_values, tau):
    exp_values = np.exp(q_values / tau)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

def softmax_policy(Qtable, state, tau):
    probabilities = softmax(Qtable[state], tau)
    action = np.random.choice(len(probabilities), p=probabilities)
    return action

def greedy_policy(Qtable, state):
    return np.argmax(Qtable[state])

def save_q_table(Qtable, filename="q_table.npy"):
    np.save(filename, Qtable)
    print(f"Q-table saved to {filename}")

def load_q_table(filename="q_table.npy"):
    if os.path.exists(filename):
        return np.load(filename)
    else:
        print(f"No saved Q-table found at {filename}. Returning a new Q-table.")
        return None

def train_episode(tau, learning_rate, env, max_steps, Qtable):
    gamma = 0.99
    obs, _ = env.reset()
    state = discretize_state(obs, 10)
    total_reward = 0

    for step in range(max_steps):
        action = softmax_policy(Qtable, state, tau)
        new_state, reward, terminated, done, _ = env.step(action)
        new_state = discretize_state(new_state, 10)

        Qtable[state][action] = Qtable[state][action] + learning_rate * (
            reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
        )

        state = new_state
        total_reward += reward

        if terminated or done:
            break

    return Qtable, total_reward, tau

def train(n_training_episodes, tau, learning_rate, tau_decay, learning_rate_decay, env, max_steps, Qtable):
    rewards_log = []
    tau_log = []
    learning_rate_log = []

    with multiprocessing.Pool() as pool:
        results = [pool.apply_async(train_episode, (tau, learning_rate, env, max_steps, Qtable)) for _ in range(n_training_episodes)]

        for result in results:
            Qtable, total_reward, tau = result.get()
            rewards_log.append(total_reward)
            tau_log.append(tau)
            learning_rate_log.append(learning_rate)

            # Decay tau and learning rate for next batch of episodes
            tau *= tau_decay  # Decay temperature after each batch
            learning_rate *= learning_rate_decay  # Decay learning rate

            # Print progress every 10 episodes
            if len(rewards_log) % 10 == 0:
                print(f"Episode {len(rewards_log)}/{n_training_episodes}, Total Reward: {total_reward:.2f}, Tau: {tau:.4f}, Learning Rate: {learning_rate:.4f}")

    save_q_table(Qtable)  # Save Q-table after training
    return Qtable, rewards_log, tau_log, learning_rate_log

def test_model(Qtable, env, n_eval):
    rewards = []
    with multiprocessing.Pool() as pool:
        results = [pool.apply_async(run_test_episode, (Qtable, env)) for _ in range(n_eval)]

        for result in results:
            reward = result.get()
            rewards.append(reward)

    return rewards

def run_test_episode(Qtable, env):
    obs, _ = env.reset()
    state = discretize_state(obs, 10)
    total_reward = 0
    done = False

    while not done:
        action = greedy_policy(Qtable, state)
        new_state, reward, terminated, done, _ = env.step(action)
        state = discretize_state(new_state, 10)
        total_reward += reward

    return total_reward

if __name__ == '__main__':
    # Initialize the environment and parameters
    env = start_env()
    action_space = env.action_space.n
    state_bins = [10] * 6  # Updated binning
    Qtable_start = load_q_table()

    # Training parameters
    n_training_episodes = 150
    max_steps = 500
    n_eval = 10  # Reduced the number of test evaluations to 10
    tau = 0.5  # Initial temperature for softmax
    tau_decay = 0.90  # Temperature decay factor
    learning_rate = 0.8  # Initial learning rate
    learning_rate_decay = 0.99  # Decay factor for learning rate

    # Train the agent
    Qtable_acrobat, training_rewards, tau_log, learning_rate_log = train(
        n_training_episodes, tau, learning_rate, tau_decay, learning_rate_decay, env, max_steps, Qtable_start
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
    plt.plot(tau_log)
    plt.xlabel("Episode")
    plt.ylabel("Tau")
    plt.title("Tau Decay")

    plt.tight_layout()
    plt.show()

    # Display evaluation results
    print("Average Test Reward:", np.mean(rewards))
    print("Standard Deviation of Test Rewards:", np.std(rewards))
