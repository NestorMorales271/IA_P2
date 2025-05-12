import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Inicialización de Pygame y configuración de la ventana
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Función para cargar una textura
def load_texture(image_path):
    texture = pygame.image.load(image_path)
    texture_data = pygame.image.tostring(texture, "RGB", True)
    width, height = texture.get_rect().size

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
    return tex_id

# Configuración de la luz y las sombras
def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))  # Posición de la luz
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))   # Luz difusa
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))  # Luz especular
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

# Dibujar un cubo con textura
def draw_textured_cube(texture_id):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glBegin(GL_QUADS)
    # Cara frontal
    glTexCoord2f(0, 0); glVertex3f(-1, -1, 1)
    glTexCoord2f(1, 0); glVertex3f(1, -1, 1)
    glTexCoord2f(1, 1); glVertex3f(1, 1, 1)
    glTexCoord2f(0, 1); glVertex3f(-1, 1, 1)
    # Cara trasera
    glTexCoord2f(0, 0); glVertex3f(-1, -1, -1)
    glTexCoord2f(1, 0); glVertex3f(1, -1, -1)
    glTexCoord2f(1, 1); glVertex3f(1, 1, -1)
    glTexCoord2f(0, 1); glVertex3f(-1, 1, -1)
    # Otras caras...
    glEnd()

# Cargar textura y configurar iluminación
texture_id = load_texture("path_to_texture.jpg")  # Reemplaza con la ruta de tu textura
setup_lighting()

# Bucle principal
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glRotatef(1, 3, 1, 1)  # Rotación del cubo
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_textured_cube(texture_id)
    pygame.display.flip()
    pygame.time.wait(10)