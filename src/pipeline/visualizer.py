"""
Módulo de visualización
"""

import cv2
import numpy as np
from pathlib import Path

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.font = config.FONT
        self.frame_count = 0
    
    def draw_detection(self, frame, det, show_distance=True):
        x1, y1, x2, y2 = det['bbox']
        cls = det['class']
        conf = det['conf']
        distance = det.get('distance', 0.0)
        
        color = self.config.CLASS_COLORS.get(cls, (255, 255, 255))
        label = self.config.CLASS_NAMES.get(cls, "Objeto")
        
        thickness = self.config.BOX_THICKNESS
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        if show_distance and distance > 0:
            text = f"{label} {conf:.2f} {distance:.1f}m"
        else:
            text = f"{label} {conf:.2f}"
        
        scale = self.config.TEXT_SCALE
        text_thickness = self.config.TEXT_THICKNESS
        (text_w, text_h), baseline = cv2.getTextSize(text, self.font, scale, text_thickness)
        
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), color, 1)
        cv2.putText(frame, text, (x1 + 3, y1 - 3), self.font, scale, color, text_thickness)
        
        if show_distance and distance > 0:
            bar_height = 2
            max_bar_width = (x2 - x1)
            normalized_dist = min(distance / self.config.MAX_DEPTH, 1.0)
            bar_width = int((1.0 - normalized_dist) * max_bar_width)
            bar_width = max(bar_width, 3)
            
            if distance < 2.0:
                bar_color = (0, 0, 255)
            elif distance < 4.0:
                bar_color = (0, 165, 255)
            elif distance < 6.0:
                bar_color = (0, 255, 255)
            else:
                bar_color = (255, 255, 0)
            
            cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_width, y2 + 4 + bar_height), bar_color, -1)
    
    def draw_point_cloud(self, frame, detections):
        h, w = frame.shape[:2]
        focal = self.config.FOCAL_LENGTH
        cx = self.config.CX if self.config.CX else w / 2
        cy = self.config.CY if self.config.CY else h / 2
        
        for det in detections:
            if 'point_cloud' not in det:
                continue
            
            points_3d, colors = det['point_cloud']
            if len(points_3d) == 0:
                continue
            
            step = max(1, len(points_3d) // 3000)
            
            for i in range(0, len(points_3d), step):
                x, y, z = points_3d[i]
                if z > 0.1:
                    u = int((x * focal / z) + cx)
                    v = int((y * focal / z) + cy)
                    if 0 <= u < w and 0 <= v < h:
                        color = (colors[i] * 255).astype(np.uint8)
                        cv2.circle(frame, (u, v), 1, color.tolist(), -1)
        
        # Guardar nube de puntos (cada 30 frames)
        self.frame_count += 1
        if self.frame_count % 30 == 0 and detections:
            self._save_pointcloud(detections)
    
    def _save_pointcloud(self, detections):
        """Guarda la nube de puntos en results/pointclouds"""
        try:
            import open3d as o3d
            from pathlib import Path
            
            all_points = []
            all_colors = []
            
            for det in detections:
                if 'point_cloud' in det:
                    pts, cols = det['point_cloud']
                    if len(pts) > 0:
                        all_points.append(pts)
                        all_colors.append(cols)
            
            if all_points:
                points = np.vstack(all_points)
                colors = np.vstack(all_colors)
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                output_path = self.config.POINT_CLOUDS_DIR / f"pointcloud_{self.frame_count:06d}.ply"
                o3d.io.write_point_cloud(str(output_path), pcd)
        except:
            pass
    
    def draw_heatmap(self, frame, depth_map, alpha=None, show_colorbar=True):
        if alpha is None:
            alpha = self.config.HEATMAP_ALPHA
        
        h, w = frame.shape[:2]
        if depth_map.shape[:2] != (h, w):
            depth_resized = cv2.resize(depth_map, (w, h))
        else:
            depth_resized = depth_map
        
        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        if depth_max - depth_min < 0.1:
            return frame
        
        depth_norm = (depth_resized - depth_min) / (depth_max - depth_min)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        if show_colorbar:
            bar_height = 200
            bar_width = 15
            bar_x = w - bar_width - 15
            bar_y = (h - bar_height) // 2
            
            color_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
            for i in range(bar_height):
                value = int(255 * (1 - i / bar_height))
                color = cv2.applyColorMap(np.array([value], dtype=np.uint8).reshape(1, 1), 
                                         cv2.COLORMAP_JET)
                color_bar[i, :] = color[0, 0]
            
            overlay[bar_y:bar_y+bar_height, bar_x:bar_x+bar_width] = color_bar
            
            font = self.font
            cv2.putText(overlay, f"{self.config.MAX_DEPTH:.0f}m", 
                       (bar_x - 5, bar_y + 12), font, 0.3, (255, 255, 255), 1)
            cv2.putText(overlay, "0m", 
                       (bar_x - 5, bar_y + bar_height + 12), font, 0.3, (255, 255, 255), 1)
        
        return overlay
    
    def draw_ui(self, frame, metrics):
        y = 30
        spacing = 22
        
        texts = [
            f"FPS: {metrics.get('fps', 0):.1f}",
            f"Dets: {metrics.get('detections', 0)}",
            f"Depth: {metrics.get('depth_time', 0):.0f}ms",
            f"Total: {metrics.get('total_time', 0):.0f}ms",
        ]
        
        for i, text in enumerate(texts):
            y_pos = y + i * spacing
            cv2.putText(frame, text, (frame.shape[1] - 150, y_pos), 
                       self.font, 0.4, self.config.UI_TEXT_COLOR, 1)
    
    def draw(self, frame, detections, depth_map=None, show_heatmap=False, 
             show_distance=True, show_clouds=False, metrics=None):
        viz_frame = frame.copy()
        
        if show_heatmap and depth_map is not None:
            viz_frame = self.draw_heatmap(viz_frame, depth_map)
            cv2.putText(viz_frame, "HEATMAP", (10, 30), 
                       self.font, 0.4, (0, 255, 255), 1)
        
        if show_clouds and detections:
            self.draw_point_cloud(viz_frame, detections)
        
        for det in detections:
            self.draw_detection(viz_frame, det, show_distance)
        
        if metrics:
            self.draw_ui(viz_frame, metrics)
            info = f"Depth: {metrics.get('depth_time', 0):.0f}ms | Fusion: {metrics.get('fusion_time', 0):.0f}ms"
            cv2.putText(viz_frame, info, (10, viz_frame.shape[0] - 12), 
                       self.font, 0.35, (180, 180, 180), 1)
        
        return viz_frame