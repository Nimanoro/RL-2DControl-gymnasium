import numpy as np
import gymnasium as gym



def discretize_state(state, state_space):

    state = [np.digitize(s, 6) for s in state]

    return state

def q_table_init():
    q_table = np.random.uniform(low=-1, high=1, size=(10, 10, 10, 10, 3))
    return q_table


def q_table_update(q_table, state, action, reward, next_state, alpha, gamma):
    q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
    return q_table

def epsilon_greedy_policy(q_table, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice([0, 1, 2])
    else:
        action = np.argmax(q_table[state])
    return action

def policy(q_table, state):
    action = np.argmax(q_table[state])
    return action


def q_learning(env, q_table, alpha, gamma, epsilon, episodes):

    for episode in range(episodes):
        state = env.reset()
        state = discretize_state(state, 10
                                 )
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(q_table, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state, [0.25, 0.5, 0.75, 1.0])
            q_table = q_table_update(q_table, state, action, reward, next_state, alpha, gamma)
            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            print('Episode {} Total Reward: {}'.format(episode, total_reward))

    return q_table


q_table = q_table_init()

env = gym.make('Acrobot-v1', render_mode="human")
gamma = 0.99
alpha = 0.01
epsilon = 0.1
episodes = 1000

q_learning(env, q_table, alpha, gamma, epsilon, episodes)

