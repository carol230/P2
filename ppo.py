# PACMAN PPO - VERSIÓN FINAL
# --------------------------------------------------------
# Juego educativo: GridWorld 7x7, Agente RL tipo Pac-Man (Proximal Policy Optimization)
# El agente aprende a resolver un laberinto con una estructura de paredes fija pero
# con frutas y venenos que cambian de lugar en cada partida.
#
# REQUISITOS:
# - pygame
# - torch (PyTorch)
# - numpy
#
# IMÁGENES REQUERIDAS (en el mismo directorio):
# agente.png, fruta.png, veneno.png, pared.png

import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import sys

# --- Hiperparámetros y Configuración Global ---
GRID_SIZE = 7
CELL_SIZE = 80
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + 80  # Espacio para la UI
STATE_SHAPE = (4, GRID_SIZE, GRID_SIZE)
ACTIONS = 4  # 0:arriba, 1:abajo, 2:izquierda, 3:derecha

# Parámetros del Agente PPO
LEARNING_RATE = 0.0003
GAMMA = 0.99
CLIP_EPSILON = 0.2
UPDATE_TIMESTEPS = 2048 # Actualizar la política cada N pasos
MODEL_PATH = "modelo_ppo_final.pth"
MAX_STEPS_PER_EPISODE = 150

# Recompensas
REWARD_STEP = -0.01
REWARD_WALL = -0.75
REWARD_FRUIT = 1.0
REWARD_LAST_FRUIT = 10.0
REWARD_POISON = -5.0

# --- Colores y Fuentes ---
COLOR_FONDO = (25, 25, 25)
COLOR_LINEAS = (40, 40, 40)
COLOR_CURSOR = (255, 255, 0)
COLOR_TEXTO = (240, 240, 240)
COLOR_INFO = (100, 200, 255)

# --- Red Neuronal Actor-Crítico (para PPO) ---
class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        c, h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * h * w, 256), nn.ReLU()
        )
        self.actor_head = nn.Sequential(nn.Linear(256, num_actions), nn.Softmax(dim=-1))
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        return self.actor_head(x), self.critic_head(x)

# --- Agente Inteligente PPO ---
class AgentePPO:
    def __init__(self, state_shape, num_actions):
        self.device = torch.device("cpu")
        self.gamma = GAMMA
        self.clip_epsilon = CLIP_EPSILON
        self.num_actions = num_actions
        self.network = ActorCritic(state_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.memory = []
        self.load()

    def almacenar(self, estado, accion, prob_accion, recompensa, terminado, valor):
        self.memory.append((estado, accion, prob_accion, recompensa, terminado, valor))

    def limpiar_memoria(self):
        self.memory = []

    def elegir_accion(self, estado, greedy=False):
        with torch.no_grad():
            estado_t = torch.tensor(estado, dtype=torch.float32, device=self.device).unsqueeze(0)
            policy, value = self.network(estado_t)
            dist = torch.distributions.Categorical(policy)
            
            if greedy:
                # En modo greedy, elegimos la acción con la probabilidad más alta
                accion_tensor = torch.argmax(policy)
            else:
                # En modo entrenamiento, muestreamos de la distribución
                accion_tensor = dist.sample()
            
            # Calculamos la probabilidad logarítmica usando el tensor de la acción
            log_prob = dist.log_prob(accion_tensor)
            # Obtenemos el valor entero de la acción para retornarlo
            accion_int = accion_tensor.item()
            
            return accion_int, log_prob, value

    def optimizar(self):
        if not self.memory: return
        states, actions, old_log_probs, rewards, dones, values = zip(*self.memory)
        
        returns = []
        discounted_reward = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                discounted_reward = 0
            discounted_reward = r + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.stack(list(old_log_probs)).detach()
        values_t = torch.cat(values).squeeze().detach()

        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        new_policy, new_values = self.network(states_t)
        dist = torch.distributions.Categorical(new_policy)
        new_log_probs = dist.log_prob(actions_t)
        
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.functional.mse_loss(new_values.squeeze(), returns_t)
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

    def save(self):
        torch.save(self.network.state_dict(), MODEL_PATH)
        print(f"Modelo PPO guardado en '{MODEL_PATH}'")

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.network.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print(f"Modelo PPO pre-entrenado cargado desde '{MODEL_PATH}'")

# --- Entorno del Juego (GridWorld) ---
class EntornoGrid:
    def __init__(self):
        self.reset_board()
        self.reset_agent()

    def reset_board(self):
        self.frutas = set()
        self.venenos = set()
        self.paredes = set()

    def reset_agent(self):
        self.agent = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
        while tuple(self.agent) in self.paredes or tuple(self.agent) in self.frutas or tuple(self.agent) in self.venenos:
             self.agent = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
        self.pasos_episodio = 0
        return self.get_state()

    def snapshot(self):
        return (set(self.frutas), set(self.venenos), set(self.paredes))

    def restore(self, snap):
        self.frutas, self.venenos, self.paredes = set(snap[0]), set(snap[1]), set(snap[2])

    def get_state(self):
        state = np.zeros(STATE_SHAPE, dtype=np.float32)
        state[0, self.agent[1], self.agent[0]] = 1.0
        for x, y in self.frutas:  state[1, y, x] = 1.0
        for x, y in self.venenos: state[2, y, x] = 1.0
        for x, y in self.paredes: state[3, y, x] = 1.0
        return state

    def paso(self, accion):
        self.pasos_episodio += 1
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][accion]
        nx, ny = self.agent[0] + dx, self.agent[1] + dy

        recompensa = REWARD_STEP
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in self.paredes):
            recompensa += REWARD_WALL
        else:
            self.agent = [nx, ny]

        pos = tuple(self.agent)
        terminado = False
        if pos in self.frutas:
            self.frutas.remove(pos)
            recompensa += REWARD_FRUIT
            if not self.frutas:
                recompensa += REWARD_LAST_FRUIT
                terminado = True
        elif pos in self.venenos:
            recompensa += REWARD_POISON
            terminado = True
        
        if self.pasos_episodio >= MAX_STEPS_PER_EPISODE:
            terminado = True

        return self.get_state(), recompensa, terminado

# --- Funciones Auxiliares ---
def cargar_imagen(nombre, fallback_color):
    try:
        img = pygame.image.load(nombre).convert_alpha()
        return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
    except pygame.error:
        print(f"Advertencia: No se pudo cargar '{nombre}'.")
        s = pygame.Surface((CELL_SIZE, CELL_SIZE)); s.fill(fallback_color)
        return s

def definir_paredes_fijas(entorno):
    """Establece la estructura de paredes predeterminada del juego."""
    entorno.paredes = {
        (1,1), (1,2), (1,3), (1,4), (1,5), 
        (5,1), (5,2), (5,3), (5,4), (5,5),
        (3,1), (3,5)
    }

def generar_objetivos_aleatorios(entorno, num_frutas, num_venenos):
    """Limpia frutas/venenos y los genera en posiciones válidas (no en paredes)."""
    entorno.frutas.clear()
    entorno.venenos.clear()
    celdas_disponibles = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) not in entorno.paredes:
                celdas_disponibles.append((x, y))
    random.shuffle(celdas_disponibles)
    for _ in range(num_frutas):
        if celdas_disponibles: entorno.frutas.add(celdas_disponibles.pop())
    for _ in range(num_venenos):
        if celdas_disponibles: entorno.venenos.add(celdas_disponibles.pop())

# --- Entrenamiento Headless ---
def entrenamiento_headless(episodios=50000, num_frutas=4, num_venenos=3, mostrar_cada=50):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    
    entorno = EntornoGrid()
    definir_paredes_fijas(entorno)
    agente = AgentePPO(state_shape=STATE_SHAPE, num_actions=ACTIONS)
    total_pasos = 0
    
    for ep in range(1, episodios + 1):
        generar_objetivos_aleatorios(entorno, num_frutas, num_venenos)
        estado = entorno.reset_agent()
        terminado, score = False, 0.0
        
        while not terminado:
            accion, log_prob, valor = agente.elegir_accion(estado)
            sig_estado, recompensa, terminado = entorno.paso(accion)
            score += recompensa
            total_pasos += 1
            agente.almacenar(estado, accion, log_prob, recompensa, terminado, valor)
            estado = sig_estado
            
            if total_pasos % UPDATE_TIMESTEPS == 0 and total_pasos > 0:
                agente.optimizar()
                agente.limpiar_memoria()

        if ep % mostrar_cada == 0:
            print(f"[Headless] Ep {ep:5d}/{episodios} | Score: {score:6.2f}")
    
    agente.save()

# --- Bucle Principal del Juego ---
def main(fps=10):
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pacman RL - PPO Final")
    clock = pygame.time.Clock()

    img_agente = cargar_imagen('agente.png', (60, 100, 255))
    img_fruta = cargar_imagen('fruta.png', (40, 200, 40))
    img_veneno = cargar_imagen('veneno.png', (255, 80, 80))
    img_pared = cargar_imagen('pared.jpg', (100, 100, 100))
    font = pygame.font.SysFont("Consolas", 24, bold=True)

    entorno = EntornoGrid()
    definir_paredes_fijas(entorno)
    agente = AgentePPO(state_shape=STATE_SHAPE, num_actions=ACTIONS)
    
    modo = "SETUP"
    cursor = [GRID_SIZE // 2, GRID_SIZE // 2]
    tablero_inicial = entorno.snapshot()
    episodio_num = 0
    puntuacion_total = 0
    pasos_desde_update = 0
    running = True

    while running:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT or (evento.type == pygame.KEYDOWN and evento.key == pygame.K_q):
                running = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_s:
                    modo = "SETUP"
                    # Al entrar a setup, restaura el mapa base (solo paredes)
                    entorno.restore(tablero_inicial)
                if evento.key == pygame.K_t:
                    modo = "TRAINING" if modo != "TRAINING" else "SETUP"
                    if modo == "TRAINING":
                        generar_objetivos_aleatorios(entorno, 4, 3) # Genera un mapa para entrenar
                        entorno.reset_agent()
                if evento.key == pygame.K_p:
                    modo = "PLAYING"
                    entorno.reset_agent()

                if modo == "SETUP":
                    if evento.key == pygame.K_UP: cursor[1] = max(0, cursor[1] - 1)
                    elif evento.key == pygame.K_DOWN: cursor[1] = min(GRID_SIZE - 1, cursor[1] + 1)
                    elif evento.key == pygame.K_LEFT: cursor[0] = max(0, cursor[0] - 1)
                    elif evento.key == pygame.K_RIGHT: cursor[0] = min(GRID_SIZE - 1, cursor[0] + 1)
                    pos = tuple(cursor)
                    if evento.key == pygame.K_f: entorno.frutas.symmetric_difference_update({pos}); entorno.venenos.discard(pos)
                    elif evento.key == pygame.K_v: entorno.venenos.symmetric_difference_update({pos}); entorno.frutas.discard(pos)
                    elif evento.key == pygame.K_c: entorno.frutas.clear(); entorno.venenos.clear()

        if modo == "TRAINING":
            estado_actual = entorno.get_state()
            accion, log_prob, valor = agente.elegir_accion(estado_actual)
            sig_estado, recompensa, terminado = entorno.paso(accion)
            puntuacion_total += recompensa
            pasos_desde_update += 1
            agente.almacenar(estado_actual, accion, log_prob, recompensa, terminado, valor)
            
            if terminado:
                print(f"Episodio {episodio_num} terminado. Puntuación: {puntuacion_total:.2f}")
                generar_objetivos_aleatorios(entorno, 4, 3)
                entorno.reset_agent()
                puntuacion_total = 0
                episodio_num += 1
                
            if pasos_desde_update >= UPDATE_TIMESTEPS:
                print(f"--- Realizando actualización de PPO ({pasos_desde_update} pasos) ---")
                agente.optimizar()
                agente.limpiar_memoria()
                agente.save()
                pasos_desde_update = 0

        elif modo == "PLAYING":
            estado_actual = entorno.get_state()
            accion, _, _ = agente.elegir_accion(estado_actual, greedy=True)
            _, _, terminado = entorno.paso(accion)
            if terminado:
                print("Episodio de juego terminado.")
                entorno.restore(tablero_inicial) # Restaura a solo paredes
                modo = "SETUP"

        pantalla.fill(COLOR_FONDO)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                pygame.draw.rect(pantalla, COLOR_LINEAS, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        for x, y in entorno.paredes: pantalla.blit(img_pared,  (x * CELL_SIZE, y * CELL_SIZE))
        for x, y in entorno.frutas:  pantalla.blit(img_fruta,  (x * CELL_SIZE, y * CELL_SIZE))
        for x, y in entorno.venenos: pantalla.blit(img_veneno, (x * CELL_SIZE, y * CELL_SIZE))
        
        if modo != "SETUP":
            pantalla.blit(img_agente, (entorno.agent[0] * CELL_SIZE, entorno.agent[1] * CELL_SIZE))
        if modo == "SETUP":
            pygame.draw.rect(pantalla, COLOR_CURSOR, (cursor[0] * CELL_SIZE, cursor[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 4)

        ui_y = GRID_SIZE * CELL_SIZE
        pygame.draw.rect(pantalla, (40,40,40), (0, ui_y, SCREEN_WIDTH, 80))
        modo_texto = font.render(f"MODO: {modo}", True, COLOR_INFO)
        pantalla.blit(modo_texto, (10, ui_y + 10))
        episodio_texto = font.render(f"Episodio: {episodio_num}", True, COLOR_TEXTO)
        pantalla.blit(episodio_texto, (250, ui_y + 10))
        controles_texto = font.render("S:Setup T:Entrenar P:Jugar Q:Salir", True, COLOR_TEXTO)
        pantalla.blit(controles_texto, (10, ui_y + 45))

        pygame.display.flip()
        clock.tick(fps)

    agente.save()
    pygame.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pac-Man RL (PPO) – Modo visual o headless")
    parser.add_argument("--headless", action="store_true", help="Entrena sin interfaz gráfica")
    parser.add_argument("--episodios", type=int, default=50000, help="Número de episodios para el entrenamiento headless")
    parser.add_argument("--fps", type=int, default=15, help="Frames por segundo en modo visual")
    args = parser.parse_args()

    if args.headless:
        entrenamiento_headless(episodios=args.episodios)
    else:
        main(fps=args.fps)