"""
Módulo de estimación de profundidad con MiDaS
CORREGIDO: distancias correctas (cerca = valor bajo, lejos = valor alto)
"""

import cv2
import torch
import numpy as np

class DepthEstimator:
    """Estimador de profundidad global (una nube por frame)"""
    
    def __init__(self, model_type, max_size, max_depth, margin, ground_ratio):
        self.max_size = max_size
        self.max_depth = max_depth
        self.margin = margin
        self.ground_ratio = ground_ratio
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Cargar MiDaS
        torch.hub.set_dir("~/.cache/torch/hub")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transformaciones
        transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ["DPT_Hybrid", "DPT_Large"]:
            self.transform = transform.dpt_transform
        else:
            self.transform = transform.small_transform
    
    def estimate_depth(self, image):
        """
        Estima mapa de profundidad de una imagen COMPLETA.
        Retorna depth_map (H, W) normalizado 0-1
        Donde: 1.0 = cerca, 0.0 = lejos
        """
        h, w = image.shape[:2]
        
        # Redimensionar
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
        else:
            image_resized = image
            new_h, new_w = h, w
        
        # Inferencia
        input_batch = self.transform(image_resized).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(new_h, new_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Normalizar: 0.0 = lejos, 1.0 = cerca
        depth_map = prediction.cpu().numpy()
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Restaurar tamaño original
        if (new_h, new_w) != (h, w):
            depth_map = cv2.resize(depth_map, (w, h))
        
        return depth_map
    
    def get_distance_from_bbox(self, depth_map, bbox):
        """
        Extrae distancia de un bbox desde el mapa de profundidad global.
        🔥 CORREGIDO: cerca = valor bajo (0-2m), lejos = valor alto (6-8m)
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        
        # 1. Recortar región del objeto con margen
        margin_top = self.margin
        margin_bottom = int((y2 - y1) * self.ground_ratio) + self.margin
        margin_left = self.margin
        margin_right = self.margin
        
        crop_y1 = max(0, y1 - margin_top)
        crop_y2 = min(h, y2 + margin_bottom)
        crop_x1 = max(0, x1 - margin_left)
        crop_x2 = min(w, x2 + margin_right)
        
        crop_depth = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if crop_depth.size == 0:
            return 0.0
        
        # 2. Región del objeto (sin márgenes)
        obj_height = y2 - y1
        obj_y1 = margin_top
        obj_y2 = margin_top + obj_height
        obj_x1 = margin_left
        obj_x2 = margin_left + (x2 - x1)
        
        # 3. Región del suelo (parte inferior del recorte)
        ground_height = int(obj_height * self.ground_ratio)
        ground_y1 = crop_depth.shape[0] - ground_height
        ground_y2 = crop_depth.shape[0]
        
        # 4. Extraer profundidades
        obj_roi = crop_depth[obj_y1:obj_y2, obj_x1:obj_x2]
        ground_roi = crop_depth[ground_y1:ground_y2, :] if ground_y2 > ground_y1 else np.array([])
        
        # 5. Calcular valores medios
        object_depth = np.mean(obj_roi) if obj_roi.size > 0 else 0.0
        ground_depth = np.mean(ground_roi) if ground_roi.size > 0 else 0.0
        
        # 6. Calcular distancia normalizada
        if ground_depth > 0 and object_depth > 0:
            # normalized = 0.0 → objeto muy cerca (profundidad alta)
            # normalized = 1.0 → objeto a la misma profundidad que el suelo
            normalized = min(object_depth / ground_depth, 1.0)
            
            # 🔥 INVERTIR: cerca = valor bajo, lejos = valor alto
            distance = (1.0 - normalized) * self.max_depth
        elif object_depth > 0:
            # Si no hay suelo visible, usar solo la profundidad del objeto
            # 🔥 INVERTIR: si object_depth es alto → cerca
            distance = (1.0 - min(object_depth, 1.0)) * self.max_depth
        else:
            distance = 0.0
        
        # Limitar a rango razonable
        return min(max(distance, 0.2), self.max_depth)