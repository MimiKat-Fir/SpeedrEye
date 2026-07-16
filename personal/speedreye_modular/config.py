"""
Configuración centralizada del sistema
"""

import cv2

class Config:
    # YOLO
    YOLO_MODEL = "yolo11n.pt"
    CONFIDENCE_THRESHOLD = 0.5
    CLASSES_INTERES = [0, 1]  # 🔥 CORREGIDO: 0=persona, 1=bicicleta
    
    # MiDaS
    DEPTH_MODEL = "MiDaS_small"
    MAX_SIZE = 320
    MAX_DEPTH = 8.0
    MARGIN = 20
    GROUND_RATIO = 0.3
    
    # Visualización
    HEATMAP_ALPHA = 0.4
    BOX_THICKNESS = 1
    TEXT_SCALE = 0.5
    TEXT_THICKNESS = 1
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    # Colores (BGR) - CORREGIDOS para clases correctas
    COLORS = {
        0: (0, 255, 0),          # Persona - Verde
        1: (0, 165, 255),        # Bicicleta - Naranja 🔥 CORREGIDO (era 2)
    }
    LABELS = {
        0: "Persona",
        1: "Bicicleta",          # 🔥 CORREGIDO (era 2)
    }
    EMOJIS = {
        0: "👤",
        1: "🚲",                 # 🔥 CORREGIDO (era 2)
    }
    
    # Colores para UI
    UI_TEXT_COLOR = (0, 255, 255)
    UI_BG_COLOR = (0, 0, 0, 0.5)
    DISTANCE_TEXT_COLOR = (255, 255, 255)
    
    # Métricas
    FPS_BUFFER_SIZE = 30