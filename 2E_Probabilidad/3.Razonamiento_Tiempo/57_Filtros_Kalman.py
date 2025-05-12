import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x0, P0):
        """
        Inicializa el Filtro de Kalman.
        A: Matriz de transición de estado
        B: Matriz de entrada de control
        H: Matriz de observación
        Q: Covarianza del ruido del proceso
        R: Covarianza del ruido de medición
        x0: Estimación inicial del estado
        P0: Estimación inicial de la covarianza
        """
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u):
        """
        Predice el siguiente estado y covarianza.
        u: Entrada de control
        """
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        """
        Actualiza la estimación del estado con una nueva medición.
        z: Medición
        """
        y = z - self.H @ self.x  # Residuo de la medición
        S = self.H @ self.P @ self.H.T + self.R  # Covarianza del residuo
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Ganancia de Kalman
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P

    def get_state(self):
        """
        Obtiene la estimación actual del estado.
        """
        return self.x

# Ejemplo de uso
if __name__ == "__main__":
    # Definir matrices para un Filtro de Kalman simple en 1D
    A = np.array([[1]])  # Matriz de transición de estado
    B = np.array([[0]])  # Matriz de entrada de control
    H = np.array([[1]])  # Matriz de observación
    Q = np.array([[1e-5]])  # Covarianza del ruido del proceso
    R = np.array([[1e-2]])  # Covarianza del ruido de medición
    x0 = np.array([[0]])  # Estimación inicial del estado
    P0 = np.array([[1]])  # Estimación inicial de la covarianza

    # Inicializar el Filtro de Kalman
    kf = KalmanFilter(A, B, H, Q, R, x0, P0)

    # Mediciones simuladas
    measurements = [1, 2, 3, 4, 5]
    for z in measurements:
        kf.predict(u=np.array([[0]]))  # Sin entrada de control
        kf.update(z=np.array([[z]]))
        print("Estimación actualizada del estado:", kf.get_state().flatten())