"""
Módulo de detección con YOLO
"""

from ultralytics import YOLO

class Detector:
    """Detector de objetos con YOLO"""
    
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.confidence = config.CONFIDENCE_THRESHOLD
        self.classes = config.CLASSES_INTERES
    
    def detect(self, frame):
        """
        Detecta objetos en un frame.
        """
        results = self.model.track(frame, persist=True, verbose=False) #cambios
        detections = []

        if results[0].boxes is not None:
            
            for box in results[0].boxes:
                if box.id is None:
                    continue
                
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) #cambio

                if conf < self.confidence:
                    continue
                if self.classes and cls not in self.classes:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': cls,
                    'conf': conf,
                    'ID': track_id #cambio
                })
        
        return detections
