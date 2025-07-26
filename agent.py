# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

# --- Constantes del Agente ---
LEARNING_RATE = 0.001
GAMMA = 0.9  # Factor de descuento para recompensas futuras
EPSILON_START = 1.0 # Probabilidad inicial de exploración
EPSILON_END = 0.01  # Probabilidad final de exploración
EPSILON_DECAY = 0.995 # Tasa de decaimiento de épsilon
MEMORY_SIZE = 10000 # Tamaño máximo de la memoria de repetición
BATCH_SIZE = 64 # Tamaño del lote para el entrenamiento
MODEL_PATH = "dqn_model.pth" # Archivo para guardar/cargar el modelo

class DQN(nn.Module):
    """ Red Neuronal Densa para aproximar la función Q. """
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """ El agente que aprende a jugar usando DQN. """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # --- Configuración del dispositivo (CPU) ---
        self.device = torch.device("cpu")
        
        # --- Redes Neuronales ---
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Poner la red objetivo en modo de evaluación

        # --- Optimizador y Memoria ---
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        
        # Cargar el modelo si ya existe
        self.load_model()

    def remember(self, state, action, reward, next_state, done):
        """ Almacena una transición (experiencia) en la memoria. """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """
        Decide una acción usando la política épsilon-greedy.
        Con probabilidad épsilon, toma una acción aleatoria (exploración).
        Con probabilidad 1-épsilon, toma la mejor acción según la red (explotación).
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def replay(self):
        """
        Entrena la red neuronal usando un lote de experiencias de la memoria.
        Este es el núcleo del algoritmo Q-learning.
        """
        if len(self.memory) < BATCH_SIZE:
            return # No entrenar si no hay suficientes experiencias

        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        # Desempaquetar el lote
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convertir a tensores de PyTorch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        # --- Cálculo de la Pérdida (Loss) ---
        # 1. Obtener los valores Q para las acciones tomadas (Q(s,a))
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # 2. Calcular el valor Q del siguiente estado (max Q(s',a'))
        # Usamos la target_net para mayor estabilidad
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # Si el episodio terminó (done=True), el valor futuro es 0
        next_q_values[dones] = 0.0

        # 3. Calcular el valor Q objetivo (y = R + gamma * max Q(s',a'))
        target_q_values = rewards + (GAMMA * next_q_values)
        
        # 4. Calcular la pérdida (Loss) entre el valor actual y el objetivo
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # --- Optimización ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # --- Actualizar Épsilon ---
        # Reducir épsilon para disminuir la exploración a lo largo del tiempo
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def update_target_net(self):
        """ Copia los pesos de la policy_net a la target_net. """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """ Guarda los pesos del modelo en un archivo. """
        torch.save(self.policy_net.state_dict(), MODEL_PATH)
        print(f"Modelo guardado en {MODEL_PATH}")

    def load_model(self):
        """ Carga los pesos del modelo desde un archivo si existe. """
        if os.path.exists(MODEL_PATH):
            self.policy_net.load_state_dict(torch.load(MODEL_PATH))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Modelo cargado desde {MODEL_PATH}")