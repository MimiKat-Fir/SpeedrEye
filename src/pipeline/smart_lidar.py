"""
Módulo de Pseudo-LiDAR inteligente
Genera nubes de puntos solo en regiones con detecciones
"""

import numpy as np
import cv2

class SmartLidarGenerator:
    """Genera nubes de puntos 3D solo alrededor de objetos detectados"""
    
    def __init__(self, config):
        self.config = config
        self.focal_length = config.FOCAL_LENGTH
        self.max_depth = config.MAX_DEPTH
        self.stride = config.PSEUDO_LIDAR_STRIDE
    
    def generate_for_detection(self, image, depth_map, detections):
        """
        Genera nube de puntos 3D solo para las regiones detectadas.
        
        Returns:
            points: Array (N, 3) de todos los puntos 3D
            colors: Array (N, 3) colores correspondientes
            detections: Lista con nubes de puntos por detección
        """
        h, w = image.shape[:2]
        cx = self.config.CX if self.config.CX else w / 2
        cy = self.config.CY if self.config.CY else h / 2
        focal = self.focal_length
        
        all_points = []
        all_colors = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            distance = det.get('distance', 5.0)
            
            # Generar puntos 3D para este bbox
            bbox_points, bbox_colors = self._generate_bbox_cloud(
                image, depth_map, (x1, y1, x2, y2), distance, cx, cy, focal
            )
            
            det['point_cloud'] = (bbox_points, bbox_colors)
            
            if len(bbox_points) > 0:
                all_points.append(bbox_points)
                all_colors.append(bbox_colors)
        
        if all_points:
            all_points = np.vstack(all_points)
            all_colors = np.vstack(all_colors)
        else:
            all_points = np.array([])
            all_colors = np.array([])
        
        return all_points, all_colors, detections
    
    def _generate_bbox_cloud(self, image, depth_map, bbox, distance, cx, cy, focal):
        """
        Genera nube de puntos 3D para un bbox específico.
        Usa la profundidad de MiDaS pero la escala con la distancia estimada.
        """
        x1, y1, x2, y2 = bbox
        margin = 10
        
        # Recortar región del bbox (con margen)
        crop_y1 = max(0, y1 - margin)
        crop_y2 = min(image.shape[0], y2 + margin)
        crop_x1 = max(0, x1 - margin)
        crop_x2 = min(image.shape[1], x2 + margin)
        
        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_depth = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if crop_img.size == 0 or crop_depth.size == 0:
            return np.array([]), np.array([])
        
        h_crop, w_crop = crop_depth.shape[:2]
        points = []
        colors = []
        
        stride = self.stride
        
        # Calcular factor de escala
        mean_depth = np.mean(crop_depth) if np.mean(crop_depth) > 0 else 1.0
        scale_factor = distance / mean_depth
        
        for v in range(0, h_crop, stride):
            for u in range(0, w_crop, stride):
                depth_val = crop_depth[v, u]
                
                if depth_val < 0.1:
                    continue
                
                # Escalar profundidad
                scaled_depth = depth_val * scale_factor
                scaled_depth = max(0.1, min(scaled_depth, self.max_depth))
                
                # Coordenadas en la imagen original
                orig_u = crop_x1 + u
                orig_v = crop_y1 + v
                
                # Proyectar a 3D
                x = (orig_u - cx) * scaled_depth / focal
                y = (orig_v - cy) * scaled_depth / focal
                z = scaled_depth
                
                points.append([x, y, z])
                colors.append(crop_img[v, u] / 255.0)
        
        points = np.array(points, dtype=np.float32) if points else np.array([])
        colors = np.array(colors, dtype=np.float32) if colors else np.array([])
        
        return points, colors