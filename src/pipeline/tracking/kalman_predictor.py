import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanPredictor:
    def __init__(self, dt=1.0):
        """
        Inicializa el filtro de Kalman para UN objeto específico.
        dt: el paso de tiempo entre frames (normalmente 1.0)
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # dim_x = 4 estados: [x, y, vx, vy]
        # dim_z = 2 mediciones: [x, y] (lo que nos da YOLO)

        # 1. Definir la matriz de transición de estado (Física del movimiento)
        self.kf.F = np.array([
            [1, 0, dt, 0 ],  # x_nuevo = x + vx * dt
            [0, 1, 0,  dt],  # y_nuevo = y + vy * dt
            [0, 0, 1,  0 ],  # vx se mantiene constante si no hay aceleración
            [0, 0, 0,  1 ]   # vy se mantiene constante
        ])

        # 2. Definir la matriz de medición (H)
        # Solo podemos MEDIR posición (x, y) de YOLO, la velocidad la INFIERE Kalman
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Ruido e incertidumbre (Ajustes de sensibilidad)
        self.kf.R *= 10.0   # Incertidumbre de la detección de YOLO
        self.kf.P *= 1000.0 # Incertidumbre inicial (al principio no sabemos la velocidad)
        self.kf.Q *= 0.01   # Ruido del proceso (cuán brusco cambia de dirección la persona)

    def update(self, center_x, center_y):
        """Paso 1: Le entregamos a Kalman la coordenada real actual que detectó YOLO."""
        measurement = np.array([center_x, center_y])
        self.kf.predict()
        self.kf.update(measurement)

    def predict_future(self, steps=10):
        """Paso 2: Le pedimos a Kalman que proyecte N pasos hacia el futuro sin nuevas mediciones."""
        future_points = []
        
        # Guardamos el estado actual para no alterarlo
        x_temp = self.kf.x.copy()
        
        # Proyectamos hacia adelante en el tiempo repetidamente
        for _ in range(steps):
            x_temp = np.dot(self.kf.F, x_temp) # Multiplicamos estado por matriz de física
            future_points.append((int(x_temp[0][0]), int(x_temp[1][0])))
            
        return future_points