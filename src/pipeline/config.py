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
    
    # Subcarpetas de resultados
    CALIBRATION_DIR = RESULTS_DIR / "calibration"
    LOGS_DIR = RESULTS_DIR / "logs"
    POINT_CLOUDS_DIR = RESULTS_DIR / "pointclouds"
    
    # Crear directorios
    for d in [CALIBRATION_DIR, LOGS_DIR, POINT_CLOUDS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Modelo YOLO
    YOLO_MODEL_PATH = str(MODELS_DIR / "yolo" / "yolo26n.pt")
    
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

    # ============================================
    # Depth Estimator
    # ============================================
    DEPTH_ENGINE = "midas"
    MAX_DEPTH = 30.0
    DEPTH_MAX_SIZE = 320

    # ============================================
    # Fusión
    # ============================================
    KNOWN_HEIGHT = 1.7
    FOCAL_LENGTH = 700
    CX = None
    CY = None
    DISTANCE_SCALE_FACTOR = 0.85

    # ============================================
    # Pseudo-LiDAR
    # ============================================
    PSEUDO_LIDAR_STRIDE = 5

    # ============================================
    # Fusión
    # ============================================
    FUSION_MARGIN = 15
    GROUND_RATIO = 0.2

    # ============================================
    # Visualización
    # ============================================
    BOX_THICKNESS = 1
    TEXT_SCALE = 0.4
    TEXT_THICKNESS = 1
    UI_TEXT_COLOR = (0, 255, 255)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    HEATMAP_ALPHA = 0.4

    # ============================================
    # Métricas
    # ============================================
    FPS_BUFFER_SIZE = 30