"""
Módulo de visualización con estilos refinados
CORREGIDO: colores y barras consistentes con la distancia real
"""

import cv2
import numpy as np

class Visualizer:
    """Visualizador con estilos personalizados"""
    
    def __init__(self, config):
        self.config = config
        self.font = config.FONT
    
    def draw_detection(self, frame, det, show_distance=True):
        """
        Dibuja una detección con estilo fino y elegante.
        🔥 CORREGIDO: barra larga = cerca, barra corta = lejos
        """
        x1, y1, x2, y2 = det['bbox']
        cls = det['class']
        conf = det['conf']
        distance = det.get('distance', 0.0)
        
        color = self.config.COLORS.get(cls, (255, 255, 255))
        label = self.config.LABELS.get(cls, f"Clase {cls}")
        emoji = self.config.EMOJIS.get(cls, "")
        
        # --- Bounding box (marco fino) ---
        thickness = self.config.BOX_THICKNESS
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # --- Etiqueta superior (clase + confianza) ---
        text = f"{emoji} {label} {conf:.2f}"
        scale = self.config.TEXT_SCALE
        text_thickness = self.config.TEXT_THICKNESS
        
        (text_w, text_h), baseline = cv2.getTextSize(text, self.font, scale, text_thickness)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 8, y1), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 8, y1), 
                     color, thickness)
        
        cv2.putText(frame, text, (x1 + 4, y1 - 4), self.font, scale, color, text_thickness)
        
        # --- Distancia (debajo del bbox) ---
        if show_distance and distance > 0:
            # Color según distancia (cerca = rojo, lejos = azul)
            if distance < 2.0:
                cat_color = (0, 0, 255)      # Rojo
                category = "🔴 Muy cerca"
            elif distance < 4.0:
                cat_color = (0, 165, 255)    # Naranja
                category = "🟠 Cerca"
            elif distance < 6.0:
                cat_color = (0, 255, 255)    # Amarillo
                category = "🟡 Media"
            else:
                cat_color = (255, 255, 0)    # Celeste
                category = "🔵 Lejos"
            
            dist_text = f"📏 {distance:.2f}m {category}"
            (d_w, d_h), d_baseline = cv2.getTextSize(dist_text, self.font, scale, text_thickness)
            
            # Fondo para distancia
            cv2.rectangle(frame, (x1, y2 + 4), (x1 + d_w + 8, y2 + d_h + 12), 
                         (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, y2 + 4), (x1 + d_w + 8, y2 + d_h + 12), 
                         cat_color, thickness)
            
            cv2.putText(frame, dist_text, (x1 + 4, y2 + d_h + 8), 
                       self.font, scale, self.config.DISTANCE_TEXT_COLOR, text_thickness)
            
            # --- Barra de distancia (CORREGIDA: larga = cerca, corta = lejos) ---
            bar_height = 3
            # 🔥 CORRECCIÓN: cerca = barra más larga, lejos = barra más corta
            # Usar inverso de la distancia normalizada
            max_bar_width = (x2 - x1)
            normalized_dist = min(distance / self.config.MAX_DEPTH, 1.0)
            bar_width = int((1.0 - normalized_dist) * max_bar_width)
            bar_width = max(bar_width, 5)  # Mínimo 5px para que se vea
            
            cv2.rectangle(frame, (x1, y2 + d_h + 18), 
                         (x1 + bar_width, y2 + d_h + 18 + bar_height), 
                         cat_color, -1)
    
    def draw_heatmap(self, frame, depth_map, alpha=None, show_colorbar=True):
        """
        Superpone mapa de calor a la imagen.
        🔥 CORRECTO: Cerca = Rojo, Lejos = Azul
        """
        if alpha is None:
            alpha = self.config.HEATMAP_ALPHA
        
        h, w = frame.shape[:2]
        
        if depth_map.shape[:2] != (h, w):
            depth_resized = cv2.resize(depth_map, (w, h))
        else:
            depth_resized = depth_map
        
        # Usar depth_resized directamente (cerca = valor alto → rojo)
        depth_uint8 = (depth_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        # Superponer
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        # Barra de color lateral
        if show_colorbar:
            bar_height = 200
            bar_width = 20
            bar_x = w - bar_width - 15
            bar_y = (h - bar_height) // 2
            
            # Generar gradiente (cerca = rojo arriba, lejos = azul abajo)
            color_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
            for i in range(bar_height):
                value = int(255 * (1 - i / bar_height))
                color = cv2.applyColorMap(np.array([value], dtype=np.uint8).reshape(1, 1), 
                                         cv2.COLORMAP_JET)
                color_bar[i, :] = color[0, 0]
            
            overlay[bar_y:bar_y+bar_height, bar_x:bar_x+bar_width] = color_bar
            
            # Etiquetas
            font = self.font
            scale = 0.35
            thickness = 1
            
            cv2.putText(overlay, "Cerca", 
                       (bar_x - 25, bar_y + 10), font, scale, (255, 255, 255), thickness)
            cv2.putText(overlay, "Lejos", 
                       (bar_x - 20, bar_y + bar_height + 12), font, scale, (255, 255, 255), thickness)
        
        return overlay, heatmap
    
    def draw_ui_text(self, frame, texts, position=(10, 30), color=None, 
                     bg_color=None, spacing=28):
        """Dibuja texto de UI (FPS, detecciones, etc.)"""
        if color is None:
            color = self.config.UI_TEXT_COLOR
        
        x, y = position
        scale = self.config.TEXT_SCALE
        thickness = self.config.TEXT_THICKNESS
        
        for i, text in enumerate(texts):
            y_pos = y + i * spacing
            
            if bg_color is not None:
                (tw, th), baseline = cv2.getTextSize(text, self.font, scale, thickness)
                cv2.rectangle(frame, (x - 5, y_pos - th - 4), 
                             (x + tw + 5, y_pos + 4), bg_color, -1)
            
            cv2.putText(frame, text, (x, y_pos), self.font, scale, color, thickness)
    
    def draw(self, frame, detections, depth_map=None, show_heatmap=False, 
             show_distance=True, show_ui=True, metrics=None):
        """Pipeline completo de dibujo"""
        viz_frame = frame.copy()
        
        # 1. Mapa de calor (si está activado)
        if show_heatmap and depth_map is not None:
            viz_frame, _ = self.draw_heatmap(viz_frame, depth_map)
            cv2.putText(viz_frame, "🔥 HEATMAP (rojo=cerca, azul=lejos)", 
                       (10, 30), self.font, 0.5, (0, 255, 255), 1)
        
        # 2. Dibujar detecciones
        for det in detections:
            self.draw_detection(viz_frame, det, show_distance)
        
        # 3. UI (FPS, detecciones, tiempos)
        if show_ui and metrics:
            h, w = viz_frame.shape[:2]
            ui_texts = [
                f"FPS: {metrics.get('fps', 0):.1f}",
                f"Dets: {len(detections)}",
                f"Depth: {metrics.get('depth_time', 0):.0f}ms",
                f"Det: {metrics.get('det_time', 0):.0f}ms",
            ]
            self.draw_ui_text(viz_frame, ui_texts, 
                             position=(w - 170, 30),
                             bg_color=(0, 0, 0))
        
        # 4. Info de procesamiento en la parte inferior
        if metrics:
            info_text = f"Total: {metrics.get('total_time', 0):.0f}ms  |  Viz: {metrics.get('viz_time', 0):.0f}ms"
            cv2.putText(viz_frame, info_text, (10, viz_frame.shape[0] - 15), 
                       self.font, 0.4, (180, 180, 180), 1)
        
        return viz_frame