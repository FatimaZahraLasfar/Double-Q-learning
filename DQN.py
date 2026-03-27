import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Paramètres du jeu
GRID_SIZE = 4
STATE_SIZE = GRID_SIZE * GRID_SIZE   # 16
ACTION_SIZE = 4                      # 0: haut, 1: bas, 2: gauche, 3: droite
GAMMA = 0.9
LEARNING_RATE = 0.01
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 1000
TARGET_UPDATE_FREQ = 10

# Déplacements possibles
MOVES = {
    0: (-1, 0),   # Haut
    1: (1, 0),    # Bas
    2: (0, -1),   # Gauche
    3: (0, 1)     # Droite
}

# ---------- Environnement GridWorld avec retour one-hot ----------
class GridWorld:
    def __init__(self, size=GRID_SIZE, start=(0, 0), goal=(3, 3), obstacle=(1, 1)):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacle = obstacle
        self.state = None

    def reset(self):
        """Réinitialise l'environnement et retourne l'état one-hot."""
        self.state = self.start
        return self._state_to_onehot(self.state)

    def step(self, action):
        """Exécute l'action, retourne (next_state_onehot, reward, done)."""
        row, col = self.state
        dr, dc = MOVES[action]
        new_row, new_col = row + dr, col + dc

        # Vérifier les limites de la grille
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            # L'action est invalide : on reste sur place et on pénalise
            new_row, new_col = row, col
            reward = -0.1
            done = False
        else:
            new_state = (new_row, new_col)
            if new_state == self.obstacle:
                reward = -1.0
                done = True
                self.state = new_state
                return self._state_to_onehot(self.state), reward, done
            elif new_state == self.goal:
                reward = 1.0
                done = True
            else:
                reward = -0.01
                done = False
            self.state = new_state

        return self._state_to_onehot(self.state), reward, done

    def _state_to_onehot(self, pos):
        """Convertit une position (ligne, colonne) en vecteur one-hot de taille 16."""
        idx = pos[0] * self.size + pos[1]
        one_hot = np.zeros(self.size * self.size, dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

# ---------- Agent Double DQN ----------
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, gamma, learning_rate,
                 epsilon, epsilon_min, epsilon_decay, batch_size, memory_size,
                 target_update_freq=10):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target()

    def _build_model(self):
        """Construit le réseau de neurones (2 couches cachées de 24 neurones)."""
        model = Sequential([
            Dense(24, activation='relu', input_shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def _update_target(self):
        """Copie les poids du réseau principal vers le réseau cible."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Sélectionne une action avec la politique ε-greedy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        """Entraîne le modèle sur un batch d'expériences (Double DQN)."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                # Sélection de la meilleure action avec le réseau principal
                q_next = self.model.predict(np.array([next_state]), verbose=0)[0]
                best_action = np.argmax(q_next)
                # Évaluation avec le réseau cible
                target_q_next = self.target_model.predict(np.array([next_state]), verbose=0)[0][best_action]
                target = reward + self.gamma * target_q_next

            # Récupération des Q-valeurs actuelles
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            q_values[action] = target

            # Mise à jour sur un exemple
            self.model.fit(np.array([state]), np.array([q_values]), epochs=1, verbose=0)

        # Décroissance d'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Mise à jour périodique du réseau cible
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self._update_target()

    def save(self, filename):
        self.model.save(filename)

# ---------- Entraînement ----------
if __name__ == "__main__":
    env = GridWorld()
    agent = DoubleDQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    )

    episode_rewards = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        for step in range(50):  # limite de 50 pas
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            agent.replay()

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Épisode {episode+1}/{EPISODES}, "
                  f"Score moyen (10 derniers): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")

    agent.save("double_dqn_model.keras")
    print("Entraînement terminé. Modèle sauvegardé.")