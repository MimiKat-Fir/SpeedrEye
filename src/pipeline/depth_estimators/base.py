"""
Clase base para estimadores de profundidad
"""

import cv2
import numpy as np

class BaseDepthEstimator:
    """Clase base que deben implementar todos los estimadores"""
    
    def __init__(self, config):
        self.config = config
        self.device = None
        self.model = None
    
    def estimate_depth(self, image):
        """
        Estima el mapa de profundidad de una imagen.
        Debe ser implementado por cada subclase.
        
        Args:
            image: Imagen RGB (numpy array)
        
        Returns:
            depth_map: Mapa de profundidad en metros (H, W)
        """
        raise NotImplementedError("Cada estimador debe implementar estimate_depth()")
    
    def preprocess_image(self, image):
        """Preprocesamiento común para todos los estimadores"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                return image
        return image