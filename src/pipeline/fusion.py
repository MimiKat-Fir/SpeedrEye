"""
Módulo de fusión YOLO + tamaño de bbox para distancia real
"""

import numpy as np

class Fusion:
    """Fusiona detecciones YOLO con tamaño del bbox para distancia"""
    
    def __init__(self, config):
        self.config = config
        self.max_depth = config.MAX_DEPTH
        self.known_height = config.KNOWN_HEIGHT
        self.focal_length = config.FOCAL_LENGTH
    
    def fuse(self, detections, depth_map, image):
        """
        Calcula distancia usando el tamaño del bounding box.
        """
        if not detections:
            return detections
        
        h, w = image.shape[:2]
        cx = self.config.CX if self.config.CX else w / 2
        cy = self.config.CY if self.config.CY else h / 2
        focal = self.focal_length
        
        # Factor de escala para ajustar distancias (calibración empírica)
        # Si las distancias son muy grandes, reducir este valor
        scale_factor = 0.85  # Ajusta según necesidad (0.5-1.0)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bbox_height = y2 - y1
            bbox_width = x2 - x1
            
            # Distancia por tamaño del bbox
            if bbox_height > 10:
                distance = (self.known_height * focal) / bbox_height
                distance = distance * scale_factor  # Aplicar factor de escala
                distance = max(0.3, min(distance, self.max_depth))
            else:
                if bbox_width > 10:
                    distance = (0.5 * focal) / bbox_width
                    distance = distance * scale_factor
                    distance = max(0.3, min(distance, self.max_depth))
                else:
                    distance = 5.0
            
            det['distance'] = distance
            
            # Posición 3D
            cx_obj = (x1 + x2) / 2
            cy_obj = (y1 + y2) / 2
            x_3d = (cx_obj - cx) * distance / focal
            y_3d = (cy_obj - cy) * distance / focal
            z_3d = distance
            
            det['position_3d'] = (x_3d, y_3d, z_3d)
        
        return detections