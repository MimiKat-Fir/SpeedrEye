"""
Configuración centralizada del pipeline
"""

import cv2
from pathlib import Path

class Config:
    # ============================================
    # Rutas
    # ============================================
    BASE_DIR = Path(__file__).parent.parent.parent  # Raíz del proyecto
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Resultados de calibracion
    CALIBRATION_DIR = RESULTS_DIR / "calibration"
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Modelo YOLO
    YOLO_MODEL_PATH = str(MODELS_DIR / "yolo" / "speedreye_best.pt")
    
    # ============================================
    # YOLO
    # ============================================
    CONFIDENCE_THRESHOLD = 0.4
    CLASSES_INTERES = [0, 1]
    CLASS_NAMES = {
        0: "Peaton",
        1: "Ciclista",
    }
    CLASS_COLORS = {
        0: (0, 255, 0),
        1: (0, 165, 255),
    }

    # Calibracion de camara
    FOCAL_LENGTH = 700.0
    CX = None
    CY = None

    # Distancia
    DISTANCE_METHOD = "geometry"
    DIRECT_DISTANCE_WEIGHTS = MODELS_DIR / "distance" / "direct_distance.pt"
    GEOMETRY_DISTANCE_WEIGHTS = MODELS_DIR / "distance" / "geometry_guided.pt"
    CLASS_HEIGHTS = {
        0: 1.70,  # Peaton
        1: 1.70,  # Ciclista con bicicleta
    }
    MIN_DISTANCE = 0.5
    MAX_DISTANCE = 60.0

    # ============================================
    # Visualización
    # ============================================
    BOX_THICKNESS = 1
    TEXT_SCALE = 0.4
    TEXT_THICKNESS = 1
    UI_TEXT_COLOR = (0, 255, 255)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # ============================================
    # Métricas
    # ============================================
    FPS_BUFFER_SIZE = 30

    # ============================================
    # VARIABLES ALERTAS DE COLISIÓN
    # ============================================
    
    # Anchura del rectángulo en la parte inferior de la imagen (px)
    ALERT_ZONE_BOTTOM_WIDTH = 450

    # Anchura en la parte superior (si quieres un trapecio)
    ALERT_ZONE_TOP_WIDTH = 180

    # Altura de la zona de seguridad (px)
    ALERT_ZONE_HEIGHT = 200

    # Número de pasos futuros de Kalman a comprobar
    ALERT_PREDICTION_STEPS = 15

    # Activar/desactivar el sistema
    ENABLE_ALERT_SYSTEM = True