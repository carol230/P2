# main.py (Versión con Modo de Juego Personalizado)
import pygame
import sys
import random
from game import GameEnvironment, CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT, FRUIT, POISON, WALL
from agent import DQNAgent

# --- Modos del Juego ---
SETUP_MODE = 0
PLAY_CUSTOM_MODE = 1 # Nuevo modo para jugar en el tablero diseñado
TRAIN_MODE = 2       # Modo para entrenamiento continuo y aleatorio

def draw_text(surface, text, pos, font_size=28, color=pygame.Color('black')):
    """Función auxiliar para dibujar texto en pantalla."""
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Agente Come-Frutas DQN")
    clock = pygame.time.Clock()

    env = GameEnvironment()
    state_size = GRID_WIDTH * GRID_HEIGHT
    action_size = 4  # Arriba, Abajo, Izquierda, Derecha
    agent = DQNAgent(state_size, action_size)

    mode = SETUP_MODE
    episode = 0
    
    cursor_pos = [GRID_HEIGHT // 2, GRID_WIDTH // 2]
    
    # Variables para el bucle de juego/entrenamiento
    state = None
    done = True

    # --- Bucle Principal del Juego ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                agent.save_model()
                running = False
            
            if event.type == pygame.KEYDOWN:
                if mode == SETUP_MODE:
                    # Mover cursor
                    if event.key == pygame.K_UP and cursor_pos[0] > 0: cursor_pos[0] -= 1
                    elif event.key == pygame.K_DOWN and cursor_pos[0] < GRID_HEIGHT - 1: cursor_pos[0] += 1
                    elif event.key == pygame.K_LEFT and cursor_pos[1] > 0: cursor_pos[1] -= 1
                    elif event.key == pygame.K_RIGHT and cursor_pos[1] < GRID_WIDTH - 1: cursor_pos[1] += 1
                    # Colocar objetos
                    elif event.key == pygame.K_f: env.place_item(cursor_pos, FRUIT)
                    elif event.key == pygame.K_v: env.place_item(cursor_pos, POISON)
                    elif event.key == pygame.K_p: env.place_item(cursor_pos, WALL)
                    
                    # --- NUEVOS CONTROLES ---
                    # Presiona ENTER para jugar en el tablero actual
                    elif event.key == pygame.K_RETURN:
                        mode = PLAY_CUSTOM_MODE
                        state = env.get_state() # Obtiene el estado del tablero que diseñaste
                        env.game_over = False
                        env.score = 0
                        done = False
                        print("--- Jugando en Nivel Personalizado ---")

                    # Presiona T para ver el entrenamiento aleatorio continuo
                    elif event.key == pygame.K_t:
                        mode = TRAIN_MODE
                        done = True # Forzará la creación de un nuevo nivel aleatorio
                        print("--- Iniciando Entrenamiento Continuo ---")
                
                # Salir de los modos de juego/entrenamiento con ESC
                elif mode == TRAIN_MODE or mode == PLAY_CUSTOM_MODE:
                    if event.key == pygame.K_ESCAPE:
                        mode = SETUP_MODE
                        print("--- Modo Configuración ---")

        # --- Lógica de Juego/Entrenamiento ---
        
        # Modo 1: Jugar en el tablero personalizado
        if mode == PLAY_CUSTOM_MODE and not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            
            if done:
                print(f"Partida personalizada terminada. Puntuación: {env.score}")
                mode = SETUP_MODE # Vuelve automáticamente al modo de configuración

        # Modo 2: Entrenamiento continuo con niveles aleatorios
        if mode == TRAIN_MODE:
            if done: 
                state = env.reset()
                for _ in range(15):
                    pos = [random.randint(0, GRID_HEIGHT-1), random.randint(0, GRID_WIDTH-1)]
                    env.place_item(pos, FRUIT)
                for _ in range(8):
                    pos = [random.randint(0, GRID_HEIGHT-1), random.randint(0, GRID_WIDTH-1)]
                    env.place_item(pos, POISON)
                print(f"Iniciando Episodio de Entrenamiento: {episode + 1}")
                done = False

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            
            if done:
                episode += 1
                agent.update_target_net()
                if episode % 25 == 0 and episode > 0:
                    agent.save_model()
                print(f"Episodio {episode} terminado. Puntuación: {env.score}. Epsilon: {agent.epsilon:.3f}")

        # --- Lógica de Dibujado (se ejecuta siempre) ---
        screen.fill((230, 230, 230))
        env.draw(screen)
        
        if mode == SETUP_MODE:
            cursor_rect = pygame.Rect(cursor_pos[1] * CELL_SIZE, cursor_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (255, 0, 0), cursor_rect, 3)
            # Instrucciones actualizadas
            draw_text(screen, "ENTER: Jugar en este nivel", (10, 150))
            draw_text(screen, "T: Ver entrenamiento automático", (10, 180))
            # ... (otras instrucciones)
        elif mode == PLAY_CUSTOM_MODE:
            draw_text(screen, "Jugando Nivel Personalizado", (10, 10), color=(50, 50, 150))
            draw_text(screen, f"Puntuación: {env.score}", (10, 40), color=(50, 50, 150))
        elif mode == TRAIN_MODE:
            draw_text(screen, f"Entrenamiento (Episodio {episode + 1})", (10, 10), color=(0, 150, 50))
            draw_text(screen, f"Puntuación: {env.score}", (10, 40), color=(0, 150, 50))

        if mode != SETUP_MODE:
            draw_text(screen, "ESC: Volver a Configuración", (10, SCREEN_HEIGHT - 30), font_size=24)

        pygame.display.flip()
        clock.tick(15)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()