import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from collections import deque

class KalmanPredictor:
    def __init__(self, fps=30.0, history_size=12):
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Matriz de transición de estado (F)
        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0,  self.dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        # Matriz de medición (H)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        self.kf.R = np.eye(2) * 0.3
        self.kf.P = np.eye(4) * 10.0

        q_x = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.05)
        self.kf.Q = np.block([
            [q_x, np.zeros((2, 2))],
            [np.zeros((2, 2)), q_x]
        ])

        # =====================================================================
        # HISTÓRICO DE POSICIONES (BUFFER HISTÓRICO DE N FRAMES)
        # =====================================================================
        # Deque guarda las últimas N posiciones (x_meters, z_meters)
        self.history = deque(maxlen=history_size)

    def update(self, x_meters, z_meters):
        """Guarda la posición en el historial y actualiza el estado de Kalman"""
        measurement = np.array([[x_meters], [z_meters]], dtype=float)
        self.kf.predict()
        self.kf.update(measurement)

        # Guardamos la posición filtrada actual en el histórico
        x_filtered = self.kf.x[0, 0]
        z_filtered = self.kf.x[1, 0]
        self.history.append((x_filtered, z_filtered))

    def predict_path_pixels(self, seconds_ahead=1.2, steps=8, fx=800.0, cx=320.0, feet_x=0, feet_y=0, cy=240.0, bbox=None):
        path = []

        # ---------------------------------------------------------------------
        # CÁLCULO DE VELOCIDAD BASADO EN EL HISTÓRICO DE MÁS DE N FRAMES
        # ---------------------------------------------------------------------
        # Si aún no tenemos suficientes frames para calcular la tendencia,
        # usamos una velocidad por defecto (flecha corta hacia el frente)
        if len(self.history) < 4:
            dx_total = 0
            dy_total = -30
        else:
            # Comparamos la posición de HACE N FRAMES con la posición ACTUAL
            x_old, z_old = self.history[0]
            x_curr, z_curr = self.history[-1]

            # Tiempo transcurrido en el histórico (en segundos)
            time_elapsed = (len(self.history) - 1) * self.dt

            # Velocidad real calculada por tendencia física (m/s)
            vx_trend = (x_curr - x_old) / time_elapsed
            vz_trend = (z_curr - z_old) / time_elapsed

            v_total = np.sqrt(vx_trend**2 + vz_trend**2)

            # Si el desplazamiento promedio en los últimos frames es muy pequeño,
            # está prácticamente parado o caminando en el sitio
            if v_total < 0.15:
                dx_total = 0
                arrow_length = 30
            else:
                # Proyección de dirección lateral basada en la trayectoria real reciente
                dir_x = np.clip(vx_trend / v_total, -1.0, 1.0)
                arrow_length = int(np.clip(v_total * 25.0, 35, 65))
                dx_total = int(dir_x * arrow_length)

            dy_total = -arrow_length

        # Generación de la línea de la flecha
        for i in range(1, steps + 1):
            factor = i / steps
            px_x = int(feet_x + (dx_total * factor))
            px_y = int(feet_y + (dy_total * factor * 0.35))
            path.append((px_x, px_y))

        return path
    

    
    def predict_path_pixels_pose(self, feet_x, feet_y, body_dx, body_dy, steps=8, arrow_length=45):
        """
        Proyecta la flecha siguiendo exactamente la orientación del torso/hombros.
        """
        path = []

        # body_dx define la inclinación lateral real del cuerpo (-1.0 izquierda, 1.0 derecha)
        dx_total = int(body_dx * arrow_length)
        dy_total = int(body_dy * arrow_length * 0.35) # Aplanado para perspectiva de suelo

        for i in range(1, steps + 1):
            factor = i / steps
            px_x = int(feet_x + (dx_total * factor))
            px_y = int(feet_y + (dy_total * factor))
            path.append((px_x, px_y))

        return path