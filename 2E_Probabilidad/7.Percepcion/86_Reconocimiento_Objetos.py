# Importamos las librerías necesarias
import cv2  # Para procesamiento de imágenes y video
import numpy as np  # Para operaciones matemáticas y de matrices

# Cargamos el modelo pre-entrenado y los archivos de configuración
# Usaremos un modelo YOLO (You Only Look Once) para reconocimiento de objetos
model_config = "yolov3.cfg"  # Archivo de configuración del modelo
model_weights = "yolov3.weights"  # Pesos pre-entrenados del modelo
labels_file = "coco.names"  # Archivo con las etiquetas de las clases

# Cargamos las etiquetas de las clases
with open(labels_file, "r") as f:
    class_labels = f.read().strip().split("\n")

# Cargamos la red neuronal con OpenCV
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Configuramos la red para usar la CPU (o GPU si está disponible)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Función para procesar la imagen y realizar el reconocimiento de objetos
def detect_objects(image):
    # Obtenemos las dimensiones de la imagen
    (H, W) = image.shape[:2]

    # Obtenemos las capas de salida del modelo YOLO
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Preprocesamos la imagen para la red YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Realizamos la detección de objetos
    layer_outputs = net.forward(output_layers)

    # Inicializamos listas para almacenar los resultados
    boxes = []  # Coordenadas de los cuadros delimitadores
    confidences = []  # Confianza de las detecciones
    class_ids = []  # IDs de las clases detectadas

    # Iteramos sobre las salidas de las capas
    for output in layer_outputs:
        for detection in output:
            # Extraemos las puntuaciones y la clase con mayor confianza
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filtramos detecciones con confianza suficiente
            if confidence > 0.5:
                # Escalamos las coordenadas del cuadro delimitador a las dimensiones originales de la imagen
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculamos las coordenadas de la esquina superior izquierda
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Guardamos los resultados
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicamos supresión no máxima para eliminar cuadros redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibujamos los cuadros delimitadores y etiquetas en la imagen
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
            label = f"{class_labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Capturamos video desde la cámara o un archivo
video_capture = cv2.VideoCapture(0)  # Cambiar a un archivo de video si es necesario

# Procesamos el video cuadro por cuadro
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Realizamos el reconocimiento de objetos en el cuadro actual
    output_frame = detect_objects(frame)

    # Mostramos el cuadro procesado
    cv2.imshow("Reconocimiento de Objetos", output_frame)

    # Salimos si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberamos los recursos
video_capture.release()
cv2.destroyAllWindows()