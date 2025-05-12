import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Importamos las librerías necesarias

# Generamos un conjunto de datos sintético para clasificación
# make_classification crea un dataset con características específicas para problemas de clasificación
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, n_classes=2, random_state=42)

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos el modelo de red neuronal utilizando Keras
model = Sequential()

# Añadimos una capa densa (fully connected) con 16 neuronas y función de activación ReLU
model.add(Dense(16, input_dim=X.shape[1], activation='relu'))

# Añadimos una segunda capa oculta con 8 neuronas y función de activación ReLU
model.add(Dense(8, activation='relu'))

# Añadimos la capa de salida con 1 neurona y función de activación sigmoide (para clasificación binaria)
model.add(Dense(1, activation='sigmoid'))

# Compilamos el modelo especificando el optimizador, la función de pérdida y la métrica
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamos el modelo con los datos de entrenamiento
# epochs define el número de iteraciones, batch_size el tamaño de los lotes
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluamos el modelo con los datos de prueba
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba: {accuracy:.2f}")

# Realizamos predicciones con el modelo entrenado
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculamos la precisión utilizando sklearn
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión calculada con sklearn: {final_accuracy:.2f}")