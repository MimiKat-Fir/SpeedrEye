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

    # Calibracion de camara
    FOCAL_LENGTH = 700.0
    CX = None
    CY = None

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
