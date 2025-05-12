import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Importar las bibliotecas necesarias
import matplotlib.pyplot as plt

# Establecer una semilla aleatoria para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Generar datos sintéticos para aprendizaje probabilístico
# Simularemos datos relacionados con estrellas en el espacio
# Características: [brillo, temperatura]
# Etiquetas: Probabilidad de ser un tipo específico de estrella (clasificación binaria)
def generate_synthetic_data(num_samples=1000):
    brightness = np.random.uniform(0, 1, num_samples)
    temperature = np.random.uniform(0, 1, num_samples)
    # Definir una relación probabilística
    probability = 1 / (1 + np.exp(-10 * (brightness - 0.5) + 5 * (temperature - 0.5)))
    labels = (probability > 0.5).astype(int)
    return np.column_stack((brightness, temperature)), labels

# Generar datos de entrenamiento y prueba
X, y = generate_synthetic_data(1000)
X_test, y_test = generate_synthetic_data(200)

# Construir un modelo de aprendizaje profundo para aprendizaje probabilístico
model = Sequential([
    Dense(64, input_dim=2, activation='relu'),  # Primera capa oculta
    Dropout(0.2),                              # Dropout para regularización
    Dense(32, activation='relu'),              # Segunda capa oculta
    Dropout(0.2),                              # Dropout para regularización
    Dense(1, activation='sigmoid')             # Capa de salida con sigmoid para probabilidad
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluar el modelo en datos de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en prueba: {test_accuracy:.2f}")

# Visualizar el proceso de entrenamiento
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Desempeño del Entrenamiento del Modelo')
plt.show()

# Predecir probabilidades para nuevos datos
new_data = np.array([[0.7, 0.8], [0.3, 0.4]])  # Ejemplo: brillo y temperatura
predictions = model.predict(new_data)
print("Probabilidades predichas para nuevos datos:", predictions)