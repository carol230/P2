# game.py
import pygame
import numpy as np
import random
import sys
import os

# --- Constantes del Juego ---
GRID_WIDTH = 7
GRID_HEIGHT = 7
CELL_SIZE = 40
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

# --- Colores ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_COLOR = (200, 200, 200)

# --- Tipos de Objetos en el Tablero ---
EMPTY = 0
AGENT = 1
FRUIT = 2
POISON = 3
WALL = 4

def load_image(file_name):
    """
    Función auxiliar para cargar una imagen.
    Maneja errores si el archivo no se encuentra y termina el programa.
    """
    if not os.path.exists(file_name):
        print(f"ERROR: No se pudo encontrar el archivo de imagen: '{file_name}'")
        print("Asegúrate de que todas las imágenes (.png) estén en la misma carpeta que el script.")
        sys.exit()
    return pygame.image.load(file_name)

class GameEnvironment:
    """
    Define el entorno del juego, incluyendo el tablero, los objetos y las reglas.
    Maneja la lógica del juego, las recompensas y el estado.
    """
    def __init__(self):
        # Cargar imágenes de forma segura y escalarlas al tamaño de la celda
        try:
            self.agent_img = pygame.transform.scale(load_image('agente.png'), (CELL_SIZE, CELL_SIZE))
            self.fruit_img = pygame.transform.scale(load_image('fruta.png'), (CELL_SIZE, CELL_SIZE))
            self.poison_img = pygame.transform.scale(load_image('veneno.png'), (CELL_SIZE, CELL_SIZE))
            self.wall_img = pygame.transform.scale(load_image('pared.jpg'), (CELL_SIZE, CELL_SIZE))
        except pygame.error as e:
            print(f"Error al cargar o procesar las imágenes con Pygame: {e}")
            sys.exit()

        self.reset()

    def reset(self):
        """
        Reinicia el estado del juego a una configuración inicial.
        Coloca al agente en una posición aleatoria y limpia el tablero.
        """
        self.grid = np.full((GRID_HEIGHT, GRID_WIDTH), EMPTY)
        
        # Posición inicial aleatoria para el agente
        self.agent_pos = [random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1)]
        self.grid[self.agent_pos[0]][self.agent_pos[1]] = AGENT
        
        self.score = 0
        self.game_over = False
        
        # Devuelve el estado inicial del tablero
        return self.get_state()

    def place_item(self, pos, item_type):
        """ Permite colocar un objeto (fruta, veneno, pared) en el tablero. """
        # Evita colocar objetos sobre el agente
        if self.grid[pos[0]][pos[1]] == AGENT:
            return False
        self.grid[pos[0]][pos[1]] = item_type
        return True

    def get_state(self):
        """
        Obtiene el estado actual del juego como una matriz numpy.
        Este estado es la entrada para la red neuronal del agente.
        Aplanamos la matriz para que sea un vector 1D.
        """
        return self.grid.flatten()

    def step(self, action):
        """
        Realiza una acción y actualiza el estado del juego.
        - action: 0=arriba, 1=abajo, 2=izquierda, 3=derecha
        - Devuelve: (next_state, reward, game_over)
        """
        if self.game_over:
            return self.get_state(), 0, self.game_over

        original_pos = list(self.agent_pos)
        
        # Mover agente basado en la acción
        if action == 0:  # Arriba
            self.agent_pos[0] -= 1
        elif action == 1:  # Abajo
            self.agent_pos[0] += 1
        elif action == 2:  # Izquierda
            self.agent_pos[1] -= 1
        elif action == 3:  # Derecha
            self.agent_pos[1] += 1

        reward = -0.05  # Pequeño castigo por cada movimiento para incentivar la eficiencia

        # 1. Colisión con los bordes del tablero
        if self.agent_pos[0] < 0 or self.agent_pos[0] >= GRID_HEIGHT or \
           self.agent_pos[1] < 0 or self.agent_pos[1] >= GRID_WIDTH:
            self.game_over = True
            reward = -1.0
            self.agent_pos = original_pos # Regresa a la posición anterior para no desaparecer
        # 2. Colisión con una pared
        elif self.grid[self.agent_pos[0]][self.agent_pos[1]] == WALL:
            self.game_over = True
            reward = -1.0
            self.agent_pos = original_pos # Choca y se queda en su sitio

        # Si no chocó, borra la posición anterior y actualiza
        if not self.game_over:
            self.grid[original_pos[0]][original_pos[1]] = EMPTY

        cell_content = self.grid[self.agent_pos[0]][self.agent_pos[1]]

        if cell_content == FRUIT:
            reward = 1.0
            self.score += 1
        elif cell_content == POISON:
            reward = -1.0
            self.score -= 1
            self.game_over = True

        self.grid[self.agent_pos[0]][self.agent_pos[1]] = AGENT
        
        if not np.any(self.grid == FRUIT):
            self.game_over = True
            if self.score > 0:
                reward += 1.0

        return self.get_state(), reward, self.game_over

    def draw(self, screen):
        """ Dibuja el estado actual del juego en la pantalla de pygame. """
        screen.fill(WHITE)
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)
                
                item = self.grid[row][col]
                if item == AGENT:
                    screen.blit(self.agent_img, rect.topleft)
                elif item == FRUIT:
                    screen.blit(self.fruit_img, rect.topleft)
                elif item == POISON:
                    screen.blit(self.poison_img, rect.topleft)
                elif item == WALL:
                    screen.blit(self.wall_img, rect.topleft)