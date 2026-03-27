# Double DQN GridWorld Agent

This project implements a Double Deep Q-Network (Double DQN) agent that learns to navigate a 4×4 grid world. The environment contains a start cell, a goal cell, and an obstacle. The agent receives a one‑hot encoded state (size 16) and chooses from four actions: up, down, left, right. The goal is to reach the target while avoiding the obstacle and staying within the grid.

## Features

- **Environment**: Custom `GridWorld` class with obstacles, boundaries, and sparse rewards.
- **Agent**: Double DQN with experience replay, target network, and ε‑greedy exploration.
- **Neural Network**: Two hidden layers with 24 neurons each, using ReLU activation and linear output.
- **Training Loop**: Runs for 1000 episodes, each limited to 50 steps, with periodic target network updates.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy

Install dependencies using pip:

```bash
pip install tensorflow numpy
```


## Usage

Run the training script:

```bash
python DQN.py
```
## Code Overview

### Environment (`GridWorld`)

- **State representation**: One‑hot vector of length 16 (one for each cell).
- **Actions**: 0=up, 1=down, 2=left, 3=right.
- **Rewards**:
  - +1.0 for reaching the goal.
  - -1.0 for hitting the obstacle (episode ends).
  - -0.01 for a normal move.
  - -0.1 for an invalid move (attempting to go outside the grid, staying in place).

### Agent (`DoubleDQNAgent`)

- **Experience replay**: Stores up to 2000 transitions in a deque.
- **Double DQN**: Uses the main network to select the best action and the target network to evaluate its Q‑value, reducing overestimation.
- **Target network update**: Copies weights from the main network every 10 training steps.
- **ε‑greedy exploration**: Starts with ε = 1.0 and decays by 0.995 per episode until a minimum of 0.01.

### Training Loop

For each episode:
1. Reset the environment.
2. For up to 50 steps (or until done):
   - Select an action using ε‑greedy.
   - Take a step, get reward and next state.
   - Store the transition in memory.
   - Perform a replay (training step) if enough experiences are available.
3. Record total reward for the episode.
4. Decay ε at the end of each episode.

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GRID_SIZE` | 4 | Size of the square grid. |
| `STATE_SIZE` | 16 | One‑hot encoded state dimension. |
| `ACTION_SIZE` | 4 | Number of actions. |
| `GAMMA` | 0.9 | Discount factor. |
| `LEARNING_RATE` | 0.01 | Learning rate for Adam optimizer. |
| `EPSILON` | 1.0 | Initial exploration rate. |
| `EPSILON_MIN` | 0.01 | Minimum exploration rate. |
| `EPSILON_DECAY` | 0.995 | Decay factor per episode. |
| `BATCH_SIZE` | 32 | Number of experiences per replay. |
| `MEMORY_SIZE` | 2000 | Maximum size of replay memory. |
| `EPISODES` | 1000 | Number of training episodes. |
| `TARGET_UPDATE_FREQ` | 10 | Steps between target network updates. |

## Output

During training, you will see lines like:

`Episode 10/1000, Average score (last 10): 0.85, Epsilon: 0.9511`

`Episode 20/1000, Average score (last 10): 0.92, Epsilon: 0.9044`


At the end, the model is saved as `double_dqn_model.keras`.
