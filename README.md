
# Acrobot Q-Learning with Softmax Policy

This project implements a **Q-learning agent** to solve the dynamic control problem of the **Acrobot environment** using **softmax policy** for action selection and state discretization.

---

## Features

### **1. Q-Learning Implementation**
- Uses a **softmax policy** for action selection, ensuring balanced exploration and exploitation.
- Discretizes continuous state space into bins for efficient Q-table representation.
- Updates Q-values with:
  - Learning rate (decayed over episodes).
  - Discount factor for future rewards.

### **2. Parallelized Training**
- Parallelized training and evaluation using **Python's multiprocessing** to optimize computational efficiency.
- Significantly reduces training time while maintaining stability.

### **3. Dynamic Hyperparameters**
- Decaying **temperature (`tau`)** for softmax and **learning rate** for improved convergence.
- Configurable bin sizes for state discretization and training parameters.

### **4. Comprehensive Evaluation**
- Runs multiple evaluation episodes post-training to ensure agent reliability.
- Logs training rewards, tau decay, and learning rate trends for visualization.

---

## Technologies Used

- **Python**: Core programming language.
- **Reinforcement Learning**: Q-learning algorithm.
- **Multiprocessing**: Parallelized training and evaluation.
- **Matplotlib**: Training and evaluation visualization.
- **Acrobot Environment**: A dynamic control problem from OpenAI Gym.

---

## Installation

### **Prerequisites**
- Python 3.8 or higher
- Install necessary packages:
  ```bash
  pip install matplotlib numpy gym
  ```

### **Running the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/acrobat-qlearning.git
   cd acrobat-qlearning
   ```
2. Run the training script:
   ```bash
   python train_acrobot.py
   ```
3. Visualize training progress and evaluation results in the terminal and output plots.

---

## Training Parameters

### **Default Parameters**
- **Episodes**: 150
- **Max Steps per Episode**: 500
- **Initial Learning Rate**: 0.8 (decayed by 0.99 per episode)
- **Initial Temperature (`tau`)**: 0.5 (decayed by 0.9 per episode)
- **Discount Factor (`gamma`)**: 0.99
- **State Bins**: 10 bins for each state dimension

### **Customizable Parameters**
- Modify parameters like the number of episodes, learning rate, and temperature decay directly in the script.

---

## Visualization

- **Training Rewards**: Plots total rewards per episode.
- **Tau Decay**: Visualizes the temperature decay over episodes.
- **Evaluation Results**: Average and standard deviation of rewards during test episodes.

---

## How It Works

1. **State Discretization**:
   - Continuous states (e.g., angular velocities, joint positions) are discretized into bins for efficient Q-table lookup.

2. **Softmax Policy**:
   - Action probabilities are computed using the softmax function based on Q-values and current temperature.

3. **Training Loop**:
   - Agent interacts with the environment, learns from rewards, and updates Q-values based on the Bellman equation.

4. **Evaluation**:
   - Trained Q-table is used to run greedy policy (exploitation only) for multiple episodes to assess performance.

---

## Results

- **Performance Metrics**:
  - Average test reward across episodes.
  - Standard deviation of test rewards to measure consistency.

- **Sample Output**:
  - Average Test Reward: `XX`
  - Standard Deviation of Test Rewards: `YY`

---

## Planned Enhancements
- Extend the implementation to handle continuous action spaces using **function approximation (e.g., neural networks)**.
- Add dynamic adjustment of bin sizes for state discretization.
- Implement policy gradient methods for comparison.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact
For inquiries or feature requests, contact [Nima Norouzi](mailto:your-email@example.com).
