"""
Estimador de profundidad con MiDaS
"""

import cv2
import torch
import numpy as np
from .base import BaseDepthEstimator

class MidasDepthEstimator(BaseDepthEstimator):
    """Estimador de profundidad usando MiDaS"""
    
    def __init__(self, config):
        super().__init__(config)
        self.max_size = config.DEPTH_MAX_SIZE
        self.max_depth = config.MAX_DEPTH
        
        print("📊 Inicializando Depth Estimator (MiDaS)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Dispositivo: {self.device}")
        
        # Cargar MiDaS
        self.model, self.transform = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✅ Depth Estimator (MiDaS) listo")
    
    def _load_model(self):
        """Carga el modelo MiDaS desde torch.hub"""
        torch.hub.set_dir("~/.cache/torch/hub")
        
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        model.eval()
        
        transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transform.small_transform
        
        return model, transform
    
    def estimate_depth(self, image):
        """Estima profundidad usando MiDaS"""
        h, w = image.shape[:2]
        
        # Reducir tamaño
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
        else:
            image_resized = image
            new_h, new_w = h, w
        
        input_batch = self.transform(image_resized).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(new_h, new_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        if (new_h, new_w) != (h, w):
            depth_map = cv2.resize(depth_map, (w, h))
        
        depth_map = depth_map * self.max_depth
        
        return depth_map