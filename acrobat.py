import gymnasium as gym


def start_env():
    return gym.make("Acrobot-v1", render_mode="human")  
