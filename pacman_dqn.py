# PACMAN DQN - CASA ABIERTA (VERSIÓN CORREGIDA Y MEJORADA)
# --------------------------------------------------------
# Juego educativo: GridWorld 7x7, Agente RL tipo Pac-Man (Deep Q-Learning)
# El agente aprende a recolectar frutas y evitar venenos de forma fluida y visible.
# Los visitantes pueden modificar el entorno con el teclado.
# El agente mejora partida a partida y su aprendizaje se guarda automáticamente.

# REQUISITOS:
# - pygame
# - torch (PyTorch)
# - numpy

# IMÁGENES REQUERIDAS (en el mismo directorio):
# agente.png, fruta.png, veneno.png, pared.png

# CONTROLES:
# - Flechas: Mover cursor en modo SETUP.
# - F: Colocar/Quitar fruta.
# - V: Colocar/Quitar veneno.
# - W: Colocar/Quitar pared.
# - C: Limpiar todo el tablero.
# - S: Entrar al modo SETUP para diseñar el nivel.
# - T: Iniciar/Pausar el ENTRENAMIENTO del agente.
# - P: Dejar que el agente JUEGUE (sin aprender, solo demuestra).
# - Q: Salir del programa.

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
STATE_SHAPE = (4, GRID_SIZE, GRID_SIZE)  # Formato PyTorch: Canales, Alto, Ancho
ACTIONS = 4  # 0:arriba, 1:abajo, 2:izquierda, 3:derecha

# Parámetros del Agente DQN
MEMORY_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.999
TARGET_UPDATE = 15  # Actualizar la red objetivo cada 15 episodios
MODEL_PATH = "modelo_dqn.pth"
MAX_STEPS_PER_EPISODE = 100 # Evita que el agente se quede atascado

# --- Colores y Fuentes ---
COLOR_FONDO = (25, 25, 25)
COLOR_LINEAS = (40, 40, 40)
COLOR_CURSOR = (255, 255, 0)
COLOR_TEXTO = (240, 240, 240)
COLOR_INFO = (100, 200, 255)

# --- Red Neuronal Convolucional (DQN) ---
# Usa capas convolucionales para "ver" el tablero, similar a como lo haría un humano.
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        # Modelo secuencial con capas convolucionales y lineales
        self.model = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * h * w, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.model(x)

# --- Memoria de Repetición de Experiencias ---
# Almacena las experiencias para que el agente pueda aprender de ellas en lotes.
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Agente Inteligente DQN ---
class AgenteDQN:
    def __init__(self, state_shape, num_actions):
        self.device = torch.device("cpu") # Optimizado para CPU
        self.policy_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net = DQN(state_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.num_actions = num_actions
        self.update_target()
        self.load() # Carga el progreso anterior si existe

    def elegir_accion(self, estado, greedy=False):
        # Si greedy es True, el agente no explora, solo usa su mejor política.
        eps = 0.0 if greedy else self.epsilon
        if random.random() < eps:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            estado_tensor = torch.tensor(estado, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.policy_net(estado_tensor)
            return q_vals.argmax().item()

    def almacenar(self, *args):
        self.memory.push(tuple(args))

    def optimizar(self):
        if len(self.memory) < BATCH_SIZE: return

        batch = self.memory.sample(BATCH_SIZE)
        estados, acciones, recompensas, sig_estados, dones = zip(*batch)

        estados_t = torch.tensor(np.array(estados), dtype=torch.float32, device=self.device)
        acciones_t = torch.tensor(acciones, dtype=torch.int64, device=self.device).unsqueeze(1)
        recompensas_t = torch.tensor(recompensas, dtype=torch.float32, device=self.device).unsqueeze(1)
        sig_estados_t = torch.tensor(np.array(sig_estados), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.policy_net(estados_t).gather(1, acciones_t)
        with torch.no_grad():
            next_q_vals = self.target_net(sig_estados_t).max(1)[0].unsqueeze(1)
            expected_q = recompensas_t + (GAMMA * next_q_vals * (1 - dones_t))

        loss = nn.functional.mse_loss(q_vals, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decaer_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self):
        torch.save(self.policy_net.state_dict(), MODEL_PATH)
        print(f"Modelo guardado en '{MODEL_PATH}'")

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.update_target()
            self.epsilon = EPSILON_END # Si ya sabe, que explore menos
            print(f"Modelo pre-entrenado cargado desde '{MODEL_PATH}'")


# --- Entorno del Juego (GridWorld) ---
class EntornoGrid:
    def __init__(self):
        self.reset_board()
        self.reset_agent()

    def reset_board(self):
        """ Limpia el tablero de frutas, venenos y paredes. """
        self.frutas = set()
        self.venenos = set()
        self.paredes = set()

    def reset_agent(self):
        """ Reinicia la posición del agente y el contador de pasos. """
        self.agent = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
        while tuple(self.agent) in self.paredes or tuple(self.agent) in self.frutas or tuple(self.agent) in self.venenos:
             self.agent = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
        self.pasos_episodio = 0
        return self.get_state()

    def snapshot(self):
        """ Guarda la configuración actual del tablero. """
        return (set(self.frutas), set(self.venenos), set(self.paredes))

    def restore(self, snap):
        """ Restaura una configuración del tablero guardada. """
        self.frutas, self.venenos, self.paredes = set(snap[0]), set(snap[1]), set(snap[2])
        self.total_frutas = len(self.frutas)

    def get_state(self):
        """ Genera el estado como una matriz 4D para la CNN. """
        state = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
        state[self.agent[1], self.agent[0], 0] = 1.0  # Canal 0: agente
        for x, y in self.frutas:  state[y, x, 1] = 1.0   # Canal 1: fruta
        for x, y in self.venenos: state[y, x, 2] = 1.0  # Canal 2: veneno
        for x, y in self.paredes: state[y, x, 3] = 1.0  # Canal 3: pared
        # CORRECCIÓN: Cambia el formato de (H, W, C) a (C, H, W) para PyTorch
        return state.transpose(2, 0, 1)

    def paso(self, accion):
        self.pasos_episodio += 1
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][accion]
        nx, ny = self.agent[0] + dx, self.agent[1] + dy

        # Penalización por moverse
        recompensa = -0.01

        # Comprueba colisiones con paredes o límites
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in self.paredes):
            recompensa = -0.2 # Penalización por chocar
        else:
            self.agent = [nx, ny]

        pos = tuple(self.agent)
        terminado = False

        # if pos in self.frutas:
        #     recompensa = 1.0
        #     self.frutas.remove(pos)
        #     if not self.frutas:  # Si no quedan frutas, termina el episodio con éxito
        #         terminado = True
        if pos in self.frutas:
            self.frutas.remove(pos)
            if not self.frutas:  # ¡Es la última fruta!
                recompensa = 20.0  # >> RECOMPENSA FINAL MUCHO MAYOR <<
                terminado = True
            else:
                recompensa = 1.0 # Recompensa estándar por una fruta intermedia
        elif pos in self.venenos:
            recompensa = -1.0
            terminado = True
        
        # El episodio también termina si excede el número de pasos
        if self.pasos_episodio >= MAX_STEPS_PER_EPISODE:
            terminado = True

        return self.get_state(), recompensa, terminado

# --- Funciones Auxiliares ---
def cargar_imagen(nombre, fallback_color):
    try:
        # CORRECCIÓN: Usar .convert_alpha() para imágenes con transparencia
        img = pygame.image.load(nombre).convert_alpha()
        return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
    except pygame.error:
        print(f"Advertencia: No se pudo cargar '{nombre}'. Usando color de fallback.")
        s = pygame.Surface((CELL_SIZE, CELL_SIZE))
        s.fill(fallback_color)
        return s

def generar_entorno_aleatorio(entorno, num_frutas, num_venenos, num_paredes):
    """
    Limpia el entorno y lo puebla con un número específico de ítems en posiciones aleatorias.
    """
    entorno.reset_board()
    entorno.pasos_episodio = 0 # CORRECCIÓN: Reinicia el contador de pasos
    
    celdas_disponibles = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
    random.shuffle(celdas_disponibles)
    
    if not celdas_disponibles: return

    pos_agente = celdas_disponibles.pop()
    entorno.agent = [pos_agente[0], pos_agente[1]]
    
    for _ in range(num_paredes):
        if not celdas_disponibles: break
        entorno.paredes.add(celdas_disponibles.pop())
    for _ in range(num_frutas):
        if not celdas_disponibles: break
        entorno.frutas.add(celdas_disponibles.pop())
    for _ in range(num_venenos):
        if not celdas_disponibles: break
        entorno.venenos.add(celdas_disponibles.pop())

# --- NUEVA FUNCIÓN UTILITARIA ---------------------------------
def run_episode(entorno, agente, entrenamiento=True):
    """
    Ejecuta un solo episodio y devuelve la puntuación obtenida.
    • Si 'entrenamiento' es False => no guarda transiciones ni optimiza.
    """
    estado = entorno.reset_agent()               # Coloca al bot
    terminado, score = False, 0.0

    while not terminado:
        accion = agente.elegir_accion(estado, greedy=not entrenamiento)
        sig_estado, recompensa, terminado = entorno.paso(accion)
        score += recompensa

        if entrenamiento:
            agente.almacenar(estado, accion, recompensa, sig_estado, terminado)
            agente.optimizar()

        estado = sig_estado

    return score

def entrenamiento_headless(episodios=10000,
                           num_frutas=4, num_venenos=3, num_paredes=6,
                           mostrar_cada=100):

    # Truco: desactiva la necesidad de abrir ventana
    import os, pygame
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()                      # Sigue haciendo falta para que no fallen imports

    entorno = EntornoGrid()
    agente  = AgenteDQN(state_shape=STATE_SHAPE, num_actions=ACTIONS)

    for ep in range(1, episodios + 1):
        generar_entorno_aleatorio(entorno, num_frutas, num_venenos, num_paredes)
        score = run_episode(entorno, agente, entrenamiento=True)

        # Decaimiento de ε y actualización de red objetivo
        agente.decaer_epsilon()
        if ep % TARGET_UPDATE == 0:
            agente.update_target()
            agente.save()

        if ep % mostrar_cada == 0 or ep == 1:
            print(f"[Headless] Ep {ep:5d}/{episodios} | Score: {score:6.2f} | ε={agente.epsilon:.3f}")

# --- Bucle Principal del Juego ---
def main(fps=10):
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pacman RL - DQN Interactivo")
    clock = pygame.time.Clock()

    # Carga de recursos
    img_agente = cargar_imagen('agente.png', (60, 100, 255))
    img_fruta = cargar_imagen('fruta.png', (40, 200, 40))
    img_veneno = cargar_imagen('veneno.png', (255, 80, 80))
    img_pared = cargar_imagen('pared.jpg', (100, 100, 100)) # CORRECCIÓN: pared.png
    font = pygame.font.SysFont("Consolas", 24, bold=True)

    # Inicialización de objetos
    entorno = EntornoGrid()
    agente = AgenteDQN(state_shape=STATE_SHAPE, num_actions=ACTIONS)
    
    # Variables de estado del juego
    modo = "SETUP" # Inicia en modo diseño
    cursor = [GRID_SIZE // 2, GRID_SIZE // 2]
    tablero_inicial = entorno.snapshot()
    episodio_num = 0
    puntuacion_total = 0
    running = True

    # ---- BUCLE PRINCIPAL ----
    while running:
        # --- 1. GESTIÓN DE EVENTOS ---
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT or (evento.type == pygame.KEYDOWN and evento.key == pygame.K_q):
                running = False
            if evento.type == pygame.KEYDOWN:
                # Cambiar de modo
                if evento.key == pygame.K_s:
                    modo = "SETUP"
                    entorno.restore(tablero_inicial)
                if evento.key == pygame.K_t:
                    if modo != "TRAINING":
                        modo = "TRAINING"
                        entorno.restore(tablero_inicial)
                        entorno.reset_agent()
                    else:
                        modo = "SETUP" # Pausar entrenamiento
                if evento.key == pygame.K_p:
                    modo = "PLAYING"
                    entorno.restore(tablero_inicial)
                    entorno.reset_agent()

                # Controles del modo SETUP
                if modo == "SETUP":
                    if evento.key == pygame.K_UP: cursor[1] = max(0, cursor[1] - 1)
                    elif evento.key == pygame.K_DOWN: cursor[1] = min(GRID_SIZE - 1, cursor[1] + 1)
                    elif evento.key == pygame.K_LEFT: cursor[0] = max(0, cursor[0] - 1)
                    elif evento.key == pygame.K_RIGHT: cursor[0] = min(GRID_SIZE - 1, cursor[0] + 1)
                    
                    pos = tuple(cursor)
                    if evento.key == pygame.K_f:
                        entorno.frutas.symmetric_difference_update({pos})
                        entorno.venenos.discard(pos); entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_v:
                        entorno.venenos.symmetric_difference_update({pos})
                        entorno.frutas.discard(pos); entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_w:
                        entorno.paredes.symmetric_difference_update({pos})
                        entorno.frutas.discard(pos); entorno.venenos.discard(pos)
                    elif evento.key == pygame.K_c:
                        entorno.reset_board()
                    
                    tablero_inicial = entorno.snapshot() # Guarda el diseño

        # --- 2. LÓGICA DEL JUEGO Y ENTRENAMIENTO (MODOS ACTIVOS) ---
        if modo in ["TRAINING", "PLAYING"]:
            estado_actual = entorno.get_state()
            
            # El agente elige la acción
            es_greedy = (modo == "PLAYING") # En modo "PLAYING", no explora
            accion = agente.elegir_accion(estado_actual, greedy=es_greedy)
            
            # El entorno ejecuta la acción
            sig_estado, recompensa, terminado = entorno.paso(accion)
            puntuacion_total += recompensa
            
            # Si está entrenando, almacena la experiencia y aprende
            if modo == "TRAINING":
                agente.almacenar(estado_actual, accion, recompensa, sig_estado, terminado)
                agente.optimizar()
            
            # Si el episodio termina, se reinicia para el siguiente
            if terminado:
                print(f"Episodio {episodio_num} terminado. Puntuación: {puntuacion_total:.2f}. Epsilon: {agente.epsilon:.3f}")
                entorno.restore(tablero_inicial)
                entorno.reset_agent()
                puntuacion_total = 0
                episodio_num += 1
                
                if modo == "TRAINING":
                    agente.decaer_epsilon()
                    if episodio_num % TARGET_UPDATE == 0:
                        agente.update_target()
                        agente.save() # Guarda el progreso

        # --- 3. RENDERIZADO (DIBUJAR EN PANTALLA) ---
        pantalla.fill(COLOR_FONDO)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                pygame.draw.rect(pantalla, COLOR_LINEAS, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        for x, y in entorno.frutas:  pantalla.blit(img_fruta,  (x * CELL_SIZE, y * CELL_SIZE))
        for x, y in entorno.venenos: pantalla.blit(img_veneno, (x * CELL_SIZE, y * CELL_SIZE))
        for x, y in entorno.paredes: pantalla.blit(img_pared,  (x * CELL_SIZE, y * CELL_SIZE))
        
        # Dibuja el agente solo si no está en modo SETUP
        if modo != "SETUP":
            pantalla.blit(img_agente, (entorno.agent[0] * CELL_SIZE, entorno.agent[1] * CELL_SIZE))

        # Dibuja el cursor en modo SETUP
        if modo == "SETUP":
            pygame.draw.rect(pantalla, COLOR_CURSOR, (cursor[0] * CELL_SIZE, cursor[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 4)

        # --- DIBUJAR UI (INFORMACIÓN Y CONTROLES) ---
        ui_y = GRID_SIZE * CELL_SIZE
        pygame.draw.rect(pantalla, (40,40,40), (0, ui_y, SCREEN_WIDTH, 80))
        
        modo_texto = font.render(f"MODO: {modo}", True, COLOR_INFO)
        pantalla.blit(modo_texto, (10, ui_y + 10))

        if modo == "TRAINING":
            eps_texto = font.render(f"Epsilon: {agente.epsilon:.2f}", True, COLOR_TEXTO)
            pantalla.blit(eps_texto, (200, ui_y + 10))
        
        episodio_texto = font.render(f"Episodio: {episodio_num}", True, COLOR_TEXTO)
        pantalla.blit(episodio_texto, (450, ui_y + 10))
        
        controles_texto = font.render("S:Setup T:Entrenar P:Jugar Q:Salir", True, COLOR_TEXTO)
        pantalla.blit(controles_texto, (10, ui_y + 45))

        pygame.display.flip()
        
        # Controlar la velocidad de la simulación
        clock.tick(10) # 10 pasos por segundo

    # --- Salir del Juego ---
    agente.save() # Guarda el último progreso al salir
    pygame.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pac‑Man RL (DQN) – Modo visual o headless")
    parser.add_argument("--headless", action="store_true",
                        help="Entrena sin interfaz gráfica")
    parser.add_argument("--episodios", type=int, default=5000,
                        help="Número de episodios para el entrenamiento headless")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames por segundo en modo visual")

    args = parser.parse_args()

    if args.headless:
        entrenamiento_headless(episodios=args.episodios)
    else:
        # main_visual acepta opcionalmente la velocidad deseada
        main(fps=args.fps)             # <- cambia tu 'main' original a 'main(fps=10)'
