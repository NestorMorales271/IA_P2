import pygame
import sys

# Inicialización de Pygame
pygame.init()

# Configuración de la pantalla
ANCHO, ALTO = 800, 600  # Dimensiones de la ventana
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Movimiento de un cuadro")

# Configuración del cuadro
cuadro_color = (255, 0, 0)  # Color del cuadro (rojo)
cuadro_tamano = 50  # Tamaño del cuadro (ancho y alto)
cuadro_x, cuadro_y = ANCHO // 2, ALTO // 2  # Posición inicial del cuadro (centro de la pantalla)
velocidad = 5  # Velocidad de movimiento del cuadro

# Bucle principal del juego
while True:
    # Manejo de eventos
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:  # Si se cierra la ventana
            pygame.quit()
            sys.exit()

    # Captura de teclas presionadas
    teclas = pygame.key.get_pressed()
    if teclas[pygame.K_UP]:  # Mover hacia arriba
        cuadro_y -= velocidad
    if teclas[pygame.K_DOWN]:  # Mover hacia abajo
        cuadro_y += velocidad
    if teclas[pygame.K_LEFT]:  # Mover hacia la izquierda
        cuadro_x -= velocidad
    if teclas[pygame.K_RIGHT]:  # Mover hacia la derecha
        cuadro_x += velocidad

    # Lógica para evitar que el cuadro salga de la pantalla
    cuadro_x = max(0, min(ANCHO - cuadro_tamano, cuadro_x))
    cuadro_y = max(0, min(ALTO - cuadro_tamano, cuadro_y))

    # Dibujar en la pantalla
    pantalla.fill((0, 0, 0))  # Limpiar la pantalla con color negro
    pygame.draw.rect(pantalla, cuadro_color, (cuadro_x, cuadro_y, cuadro_tamano, cuadro_tamano))  # Dibujar el cuadro
    pygame.display.flip()  # Actualizar la pantalla

    # Control de la velocidad del juego
    pygame.time.Clock().tick(60)  # Limitar a 60 FPS