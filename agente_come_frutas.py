# agente_come_frutas.py (Versión con imágenes para fruta y veneno)

import pygame
import numpy as np
import random
import time
import os

# --- CONSTANTES DE CONFIGURACIÓN DEL JUEGO Y LA PANTALLA ---
GRID_WIDTH = 15
GRID_HEIGHT = 15
CELL_SIZE = 40
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Colores (formato RGB)
COLOR_FONDO = (25, 25, 25)
COLOR_LINEAS = (40, 40, 40)
COLOR_AGENTE = (60, 100, 255)
# Los colores de fruta y veneno ya no son necesarios
COLOR_PARED = (80, 80, 80)
COLOR_TEXTO = (230, 230, 230)
COLOR_CURSOR = (255, 255, 0)

# --- PARÁMETROS DEL APRENDIZAJE POR REFUERZO (Q-LEARNING) ---
RECOMPENSA_FRUTA = 100
CASTIGO_VENENO = -100
RECOMPENSA_MOVIMIENTO = -0.1
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
NUM_EPISODIOS_ENTRENAMIENTO = 20000

class AgenteQLearning:
    def __init__(self, num_estados, num_acciones):
        self.num_acciones = num_acciones
        self.q_table = np.zeros((num_estados[0], num_estados[1], num_acciones))
        self.epsilon = EPSILON

    def elegir_accion(self, estado):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_acciones - 1)
        else:
            return np.argmax(self.q_table[estado])

    def actualizar_q_table(self, estado, accion, recompensa, nuevo_estado):
        valor_antiguo = self.q_table[estado][accion]
        valor_futuro_maximo = np.max(self.q_table[nuevo_estado])
        nuevo_q = valor_antiguo + ALPHA * (recompensa + GAMMA * valor_futuro_maximo - valor_antiguo)
        self.q_table[estado][accion] = nuevo_q

    def decaimiento_epsilon(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

class EntornoGrid:
    def __init__(self):
        self.agente_pos = (0, 0)
        self.frutas = set()
        self.venenos = set()
        self.paredes = set()
        self.reset_a_configuracion_inicial()

    def reset_a_configuracion_inicial(self):
        self.agente_pos = (0, 0)
        return self.agente_pos

    def limpiar_entorno(self):
        self.frutas.clear()
        self.venenos.clear()
        self.paredes.clear()

    # <-- MODIFICADO: La función ahora recibe el modo de juego
    def step(self, accion, modo_juego):
        x, y = self.agente_pos
        if accion == 0: y -= 1
        elif accion == 1: y += 1
        elif accion == 2: x -= 1
        elif accion == 3: x += 1

        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT or (x, y) in self.paredes:
            return self.agente_pos, RECOMPENSA_MOVIMIENTO, False

        self.agente_pos = (x, y)
        nuevo_estado = self.agente_pos
        terminado = False

        if nuevo_estado in self.frutas:
            recompensa = RECOMPENSA_FRUTA
            self.frutas.remove(nuevo_estado)
            
            # El episodio termina SÓLO si ya no quedan frutas.
            if not self.frutas:
                terminado = True

                    
        elif nuevo_estado in self.venenos:
            recompensa = CASTIGO_VENENO
            terminado = True # Tocar un veneno siempre termina el juego
        else:
            recompensa = RECOMPENSA_MOVIMIENTO

        return nuevo_estado, recompensa, terminado

    # La función dibujar no cambia
    def dibujar(self, pantalla, modo_juego, cursor_pos, img_fruta, img_veneno,img_pared, img_agente):
        pantalla.fill(COLOR_FONDO)
        
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        for pared in self.paredes:
            pantalla.blit(img_pared, (pared[0]*CELL_SIZE, pared[1]*CELL_SIZE))

        for fruta in self.frutas:
            pantalla.blit(img_fruta, (fruta[0]*CELL_SIZE, fruta[1]*CELL_SIZE))

        for veneno in self.venenos:
            pantalla.blit(img_veneno, (veneno[0]*CELL_SIZE, veneno[1]*CELL_SIZE))

        pantalla.blit(img_agente, (self.agente_pos[0]*CELL_SIZE, self.agente_pos[1]*CELL_SIZE))

        if modo_juego == "SETUP":
            cursor_rect = pygame.Rect(cursor_pos[0]*CELL_SIZE, cursor_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(pantalla, COLOR_CURSOR, cursor_rect, 3) 

        font = pygame.font.Font(None, 24)
        texto_modo = font.render(f"Modo: {modo_juego}", True, COLOR_TEXTO)
        controles_setup = font.render("SETUP: Mover con flechas. F=Fruta, V=Veneno, W=Pared. 'C' para limpiar.", True, COLOR_TEXTO)
        controles_run = font.render("'T' para Entrenar, 'P' para Jugar, 'S' para Setup.", True, COLOR_TEXTO)
        
        pantalla.blit(texto_modo, (10, SCREEN_HEIGHT + 5))
        pantalla.blit(controles_setup, (10, SCREEN_HEIGHT + 30))
        pantalla.blit(controles_run, (10, SCREEN_HEIGHT + 55))

# --- FUNCIÓN PRINCIPAL DEL JUEGO ---

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
    pygame.display.set_caption("Agente Come-Frutas 🍓 vs ☠️ (Q-Learning)")
    
    # --- Carga de imágenes ---
    # Cargar imagen de la fruta
    try:
        ruta_fruta = os.path.join(os.path.dirname(__file__), 'fruta.png')
        img_fruta_original = pygame.image.load(ruta_fruta).convert_alpha()
        img_fruta = pygame.transform.scale(img_fruta_original, (CELL_SIZE, CELL_SIZE))
    except pygame.error:
        print("Advertencia: No se encontró 'fruta.png'. Se usará un cuadrado verde.")
        img_fruta = pygame.Surface((CELL_SIZE, CELL_SIZE))
        img_fruta.fill((40, 200, 40))

    # <-- NUEVO: Cargar imagen del veneno
    try:
        ruta_veneno = os.path.join(os.path.dirname(__file__), 'veneno.png')
        img_veneno_original = pygame.image.load(ruta_veneno).convert_alpha()
        img_veneno = pygame.transform.scale(img_veneno_original, (CELL_SIZE, CELL_SIZE))
    except pygame.error:
        print("Advertencia: No se encontró 'veneno.png'. Se usará un cuadrado rojo.")
        img_veneno = pygame.Surface((CELL_SIZE, CELL_SIZE))
        img_veneno.fill((255, 50, 50))
    
    try:
        ruta_pared = os.path.join(os.path.dirname(__file__), 'pared.jpg')
        img_pared_original = pygame.image.load(ruta_pared).convert_alpha()
        img_pared = pygame.transform.scale(img_pared_original, (CELL_SIZE, CELL_SIZE))
    except pygame.error:
        print("Advertencia: No se encontró 'pared.jpg'. Se usará un cuadrado verde.")
        img_pared = pygame.Surface((CELL_SIZE, CELL_SIZE))
        img_pared.fill((40, 200, 40))

    try:
        ruta_agente = os.path.join(os.path.dirname(__file__), 'slime.png')
        img_agente_original = pygame.image.load(ruta_agente).convert_alpha()
        img_agente = pygame.transform.scale(img_agente_original, (CELL_SIZE, CELL_SIZE))
    except pygame.error:
        print("Advertencia: No se encontró 'slime.png'. Se usará un cuadrado verde.")
        img_agente = pygame.Surface((CELL_SIZE, CELL_SIZE))
        img_agente.fill((40, 200, 40))

    entorno = EntornoGrid()
    agente = AgenteQLearning(num_estados=(GRID_HEIGHT, GRID_WIDTH), num_acciones=4)
    
    reloj = pygame.time.Clock()
    corriendo = True
    modo_juego = "SETUP"
    cursor_pos = [0, 0]

    frutas_iniciales = entorno.frutas.copy()
    venenos_iniciales = entorno.venenos.copy()

    while corriendo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_t:
                    if modo_juego != "TRAINING":
                        print("--- INICIANDO ENTRENAMIENTO ---")
                        modo_juego = "TRAINING"
                        agente = AgenteQLearning(num_estados=(GRID_HEIGHT, GRID_WIDTH), num_acciones=4)
                        pantalla.fill(COLOR_FONDO)
                        font = pygame.font.Font(None, 50)
                        texto_entrenando = font.render("Entrenando...", True, COLOR_TEXTO)
                        rect = texto_entrenando.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                        pantalla.blit(texto_entrenando, rect)
                        pygame.display.flip()
                        
                        # Bucle de entrenamiento
                        for episodio in range(NUM_EPISODIOS_ENTRENAMIENTO):
                            entorno.frutas = frutas_iniciales.copy()
                            entorno.venenos = venenos_iniciales.copy()
                            estado = entorno.reset_a_configuracion_inicial()
                            terminado = False
                            while not terminado:
                                accion = agente.elegir_accion(estado)
                                # <-- MODIFICADO: Se pasa el modo "TRAINING"
                                nuevo_estado, recompensa, terminado = entorno.step(accion, "TRAINING")
                                agente.actualizar_q_table(estado, accion, recompensa, nuevo_estado)
                                estado = nuevo_estado
                            agente.decaimiento_epsilon()
                            if (episodio + 1) % 1000 == 0:
                                print(f"Episodio: {episodio + 1}/{NUM_EPISODIOS_ENTRENAMIENTO}, Epsilon: {agente.epsilon:.4f}")
                        
                        print("--- ENTRENAMIENTO COMPLETADO ---")
                        # --- NUEVO: preparar un tablero limpio para la demostración ----------
                        entorno.frutas  = frutas_iniciales.copy()
                        entorno.venenos = venenos_iniciales.copy()
                        entorno.reset_a_configuracion_inicial()
                        agente.epsilon  = 0                # solo explotación
# --------------------------------------------------------------------
                        modo_juego = "PLAYING"

                elif evento.key == pygame.K_p:
                    print("--- MODO JUEGO (AGENTE ENTRENADO) ---")
                    modo_juego = "PLAYING"
                    entorno.frutas = frutas_iniciales.copy()
                    entorno.venenos = venenos_iniciales.copy()
                    entorno.reset_a_configuracion_inicial()
                    agente.epsilon = 0

                elif evento.key == pygame.K_s:
                    print("--- MODO SETUP ---")
                    modo_juego = "SETUP"
                
                if modo_juego == "SETUP":
                    if evento.key == pygame.K_UP:
                        cursor_pos[1] = max(0, cursor_pos[1] - 1)
                    elif evento.key == pygame.K_DOWN:
                        cursor_pos[1] = min(GRID_HEIGHT - 1, cursor_pos[1] + 1)
                    elif evento.key == pygame.K_LEFT:
                        cursor_pos[0] = max(0, cursor_pos[0] - 1)
                    elif evento.key == pygame.K_RIGHT:
                        cursor_pos[0] = min(GRID_WIDTH - 1, cursor_pos[0] + 1)
                    pos_celda = tuple(cursor_pos)
                    if evento.key == pygame.K_f:
                        if pos_celda in entorno.frutas:
                            entorno.frutas.remove(pos_celda)
                        else:
                            entorno.frutas.add(pos_celda)
                            entorno.venenos.discard(pos_celda)
                            entorno.paredes.discard(pos_celda)
                    elif evento.key == pygame.K_v:
                        if pos_celda in entorno.venenos:
                            entorno.venenos.remove(pos_celda)
                        else:
                            entorno.venenos.add(pos_celda)
                            entorno.frutas.discard(pos_celda)
                            entorno.paredes.discard(pos_celda)
                    elif evento.key == pygame.K_w:
                        if pos_celda in entorno.paredes:
                            entorno.paredes.remove(pos_celda)
                        else:
                            entorno.paredes.add(pos_celda)
                            entorno.frutas.discard(pos_celda)
                            entorno.venenos.discard(pos_celda)
                    elif evento.key == pygame.K_c:
                        print("--- TABLERO LIMPIO ---")
                        entorno.limpiar_entorno()
                    frutas_iniciales = entorno.frutas.copy()
                    venenos_iniciales = entorno.venenos.copy()

        # Lógica del juego en modo PLAYING
        if modo_juego == "PLAYING":
            if entorno.frutas:
                estado = entorno.agente_pos
                accion = agente.elegir_accion(estado)
                # <-- MODIFICADO: Se pasa el modo "PLAYING"
                _, _, terminado = entorno.step(accion, "PLAYING")
                if terminado:
                    # Esto solo ocurrirá si se come la última fruta o se toca un veneno
                    modo_juego = "SETUP"
                    if not entorno.frutas:
                        print("¡Todas las frutas recolectadas! Volviendo a modo SETUP.")
                    else:
                        print("Juego terminado (veneno). Volviendo a modo SETUP.")
                time.sleep(0.1)
            else:
                # Si el juego empieza sin frutas o ya se comieron todas
                modo_juego = "SETUP"
                print("¡Todas las frutas recolectadas! Volviendo a modo SETUP.")
        
        if modo_juego != "TRAINING":
            pantalla_con_info = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
            pantalla_con_info.fill(COLOR_FONDO)
            entorno.dibujar(pantalla_con_info, modo_juego, tuple(cursor_pos), img_fruta, img_veneno, img_pared, img_agente)
            pantalla.blit(pantalla_con_info, (0,0))
            pygame.display.flip()
        
        reloj.tick(60)

    pygame.quit()
if __name__ == '__main__':
    main()