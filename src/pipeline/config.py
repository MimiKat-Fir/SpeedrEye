"""
Configuración centralizada del pipeline
"""

import cv2
from pathlib import Path

class Config:
    # ============================================
    # Rutas
    # ============================================
    BASE_DIR = Path(__file__).parent.parent.parent
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    
    CALIBRATION_DIR = RESULTS_DIR / "calibration"
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Modelo YOLO
    YOLO_MODEL_PATH = str(MODELS_DIR / "yolo" / "speedreye_kitti_v2.pt")
    
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
    # Calibracion de camara
    # ============================================
    FOCAL_LENGTH = 700.0
    CX = None
    CY = None
    ENABLE_SCENE_RECALIBRATION = False   # <- nuevo, desactivado por defecto
    SCENE_RECALIBRATION_INTERVAL = 150   # si se activa, cada cuántos frames (no cada 10)
    HORIZON_MAX_JUMP_RATIO = 0.08        # % de la altura del frame; rechaza saltos mayores
    SAVE_CALIBRATION_REPORT = True

    # ============================================
    # Distancia
    # ============================================
    DISTANCE_METHOD = "geometry"
    DIRECT_DISTANCE_WEIGHTS = MODELS_DIR / "distance" / "direct_distance.pt"
    GEOMETRY_DISTANCE_WEIGHTS = MODELS_DIR / "distance" / "geometry_guided_v1.pt"
    CLASS_HEIGHTS = {
        0: 1.70,
        1: 1.70,
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
    # ALERTAS DE COLISIÓN (configuración por defecto)
    # ============================================
    ALERT_ZONE_BOTTOM_WIDTH = 450
    ALERT_ZONE_TOP_WIDTH = 180
    ALERT_ZONE_HEIGHT = 200
    ALERT_PREDICTION_STEPS = 15
    ENABLE_ALERT_SYSTEM = True

    # ============================================
    # RENDIMIENTO (optimizaciones)
    # ============================================
    # Cada cuántos frames actualizar la pose (0 = desactivado)
    POSE_FRAME_SKIP = 3
    
    # Número mínimo de frames en histórico para predicción
    MIN_HISTORY_FOR_PREDICTION = 3
    
    # Segundos de predicción
    PREDICTION_SECONDS = 1.0
    
    # Pasos de predicción
    PREDICTION_STEPS = 6
    
    # Longitud de la flecha
    ARROW_LENGTH = 50