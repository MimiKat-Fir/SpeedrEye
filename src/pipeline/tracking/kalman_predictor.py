import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from collections import deque


class KalmanPredictor:
    def __init__(self, fps=30.0, history_size=8,
                 direction_smoothing=0.35, min_speed_threshold=0.15):
        """
        direction_smoothing: peso del EMA para suavizar la dirección (0-1, más alto = reacciona más rápido)
        min_speed_threshold: velocidad mínima (m/s) para considerar que el objeto se está moviendo
        """
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.direction_smoothing = direction_smoothing
        self.min_speed_threshold = min_speed_threshold
        self.direction = None  # vector unitario (x, z) suavizado

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0,  self.dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

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

        self.history = deque(maxlen=history_size)

    def update(self, x_meters, z_meters):
        """Actualiza el filtro con una nueva medición (posición en metros)."""
        measurement = np.array([[x_meters], [z_meters]], dtype=float)
        self.kf.predict()
        self.kf.update(measurement)
        self.history.append((self.kf.x[0, 0], self.kf.x[1, 0]))

    def get_motion_state(self):
        """Velocidad y dirección de movimiento, tal como las estima el propio Kalman."""
        vx = self.kf.x[2, 0]
        vz = self.kf.x[3, 0]
        speed = float(np.hypot(vx, vz))
        if speed > 1e-6:
            direction = np.array([vx, vz]) / speed
        else:
            direction = np.array([0.0, 0.0])
        return vx, vz, speed, direction

    def _smooth_direction(self, new_dir):
        if self.direction is None:
            self.direction = new_dir
        else:
            blended = (1 - self.direction_smoothing) * self.direction + self.direction_smoothing * new_dir
            norm = np.linalg.norm(blended)
            self.direction = blended / norm if norm > 1e-6 else self.direction
        return self.direction

    def predict_path_pixels(self, fx, center_x, center_y, steps=6,
                             body_orientation=None, pose_weight=0.4,
                             seconds_ahead=1.0):
        """
        Predicción unificada de trayectoria futura, devuelta en píxeles.

        Fusiona la tendencia de movimiento (Kalman) con la orientación corporal
        (pose), cuando está disponible. Devuelve None si no hay histórico
        suficiente o si el objeto está prácticamente quieto (evita dibujar y
        evaluar alertas sobre objetos estáticos).
        """
        if len(self.history) < 3:
            return None

        vx, vz, speed, motion_dir = self.get_motion_state()

        if speed < self.min_speed_threshold:
            self.direction = None  # reset: al retomar movimiento no arrastra dirección vieja
            return None

        if body_orientation is not None:
            bx, by = body_orientation
            # body_orientation viene en coords de imagen (dx: derecha, dy: hacia arriba)
            body_vec = np.array([bx, -by])
            norm = np.linalg.norm(body_vec)
            if norm > 1e-6:
                body_vec = body_vec / norm
                fused = (1 - pose_weight) * motion_dir + pose_weight * body_vec
                fused_norm = np.linalg.norm(fused)
                if fused_norm > 1e-6:
                    motion_dir = fused / fused_norm

        direction = self._smooth_direction(motion_dir)
        z_ref = max(self.history[-1][1], 0.5)  # distancia actual, evita división por 0

        path = []
        for i in range(1, steps + 1):
            t = seconds_ahead * (i / steps)
            dx_m = direction[0] * speed * t
            dz_m = direction[1] * speed * t
            # Proyección pinhole para el desplazamiento lateral
            px_x = int(center_x + (dx_m * fx) / z_ref)
            # Aproximación vertical (acercarse/alejarse desplaza el punto en pantalla)
            px_y = int(center_y - (dz_m * fx) / z_ref * 0.15)
            path.append((px_x, px_y))

        return path