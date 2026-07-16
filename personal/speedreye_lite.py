#!/usr/bin/env python3
"""
SpeedrEye Lite - Nube Global con Mapa de Calor
Muestra la profundidad como mapa de calor superpuesto a la imagen
"""

import cv2
import torch
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

# ============================================
# CONFIGURACIÓN
# ============================================

CLASSES_INTERES = [0, 2]        # 0=persona, 2=bicicleta
CONFIDENCE_THRESHOLD = 0.5
MAX_DEPTH = 8.0                 # Profundidad máxima (unidades relativas)
DEPTH_MODEL = "MiDaS_small"     # "MiDaS_small" (rápido) o "DPT_Hybrid" (preciso)
MAX_SIZE = 320                  # Tamaño máximo para MiDaS
MARGIN = 20                     # Margen para extraer profundidad
GROUND_RATIO = 0.3              # Proporción de suelo en el recorte

# ============================================
# COLORES
# ============================================

COLORS = {
    0: (0, 255, 0),      # Persona - Verde
    2: (0, 165, 255),    # Bicicleta - Naranja
}

LABELS = {
    0: "👤 Persona",
    2: "🚲 Bicicleta",
}

# ============================================
# CLASE: ESTIMADOR DE PROFUNDIDAD
# ============================================

class GlobalDepthEstimator:
    """
    Estimador de profundidad que genera UNA nube de puntos por frame.
    Luego extrae distancias de cada detección desde la nube.
    """
    
    def __init__(self):
        print("📊 Cargando modelo de profundidad (modo global)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Cargar MiDaS
        torch.hub.set_dir("~/.cache/torch/hub")
        self.model = torch.hub.load("intel-isl/MiDaS", DEPTH_MODEL, trust_repo=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transformaciones
        transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if DEPTH_MODEL in ["DPT_Hybrid", "DPT_Large"]:
            self.transform = transform.dpt_transform
        else:
            self.transform = transform.small_transform
        
        print(f"✅ DepthEstimator listo (dispositivo: {self.device})")
    
    def estimate_depth(self, image):
        """
        Estima mapa de profundidad de una imagen COMPLETA.
        Retorna depth_map (H, W) normalizado 0-1
        """
        h, w = image.shape[:2]
        
        # Redimensionar
        if max(h, w) > MAX_SIZE:
            scale = MAX_SIZE / max(h, w)
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
        
        # Normalizar
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
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        
        # Recortar región del objeto (con margen)
        margin_top = MARGIN
        margin_bottom = int((y2 - y1) * GROUND_RATIO) + MARGIN
        margin_left = MARGIN
        margin_right = MARGIN
        
        crop_y1 = max(0, y1 - margin_top)
        crop_y2 = min(h, y2 + margin_bottom)
        crop_x1 = max(0, x1 - margin_left)
        crop_x2 = min(w, x2 + margin_right)
        
        crop_depth = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if crop_depth.size == 0:
            return 0
        
        # Región del objeto (sin márgenes)
        obj_height = y2 - y1
        obj_y1 = margin_top
        obj_y2 = margin_top + obj_height
        obj_x1 = margin_left
        obj_x2 = margin_left + (x2 - x1)
        
        # Región del suelo (parte inferior)
        ground_height = int(obj_height * GROUND_RATIO)
        ground_y1 = crop_depth.shape[0] - ground_height
        ground_y2 = crop_depth.shape[0]
        
        # Obtener profundidades
        obj_roi = crop_depth[obj_y1:obj_y2, obj_x1:obj_x2]
        ground_roi = crop_depth[ground_y1:ground_y2, :] if ground_y2 > ground_y1 else np.array([])
        
        object_depth = np.mean(obj_roi) if obj_roi.size > 0 else 0
        ground_depth = np.mean(ground_roi) if ground_roi.size > 0 else 0
        
        # Calcular distancia
        if ground_depth > 0:
            normalized = object_depth / ground_depth if ground_depth > 0 else 0
            distance = (1.0 - min(normalized, 1.0)) * MAX_DEPTH
        else:
            distance = object_depth * MAX_DEPTH
        
        return min(max(distance, 0.2), MAX_DEPTH)
    
    @staticmethod
    def create_heatmap_overlay(frame, depth_map, alpha=0.5):
        """
        Crea un mapa de calor superpuesto a la imagen.
        
        Args:
            frame: Imagen original (BGR)
            depth_map: Mapa de profundidad (H, W) normalizado 0-1
            alpha: Transparencia (0.0 - 1.0)
        
        Returns:
            Imagen con superposición de mapa de calor
        """
        # Asegurar que depth_map tiene el mismo tamaño que frame
        h, w = frame.shape[:2]
        if depth_map.shape[:2] != (h, w):
            depth_resized = cv2.resize(depth_map, (w, h))
        else:
            depth_resized = depth_map
        
        # Convertir a uint8 (0-255)
        depth_uint8 = (depth_resized * 255).astype(np.uint8)
        
        # Aplicar mapa de color (plasma/jet)
        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
        
        # Superponer con transparencia
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return overlay, heatmap

# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SpeedrEye Lite - Nube Global con Mapa de Calor")
    parser.add_argument("--video", type=str, help="Ruta a video (o 0 para cámara)")
    parser.add_argument("--camera", action="store_true", help="Usar cámara web")
    parser.add_argument("--no_depth", action="store_true", help="Desactivar profundidad")
    parser.add_argument("--show_heatmap", action="store_true", help="Mostrar mapa de calor superpuesto")
    parser.add_argument("--alpha", type=float, default=0.4, help="Transparencia del mapa de calor (0-1)")
    
    args = parser.parse_args()
    
    # Determinar fuente
    if args.camera:
        video_path = 0
    elif args.video:
        video_path = args.video
    else:
        video_path = 0
    
    # ============================================
    # INICIALIZAR
    # ============================================
    
    print("\n" + "="*50)
    print("🚗 SPEEDREYE LITE - Nube de Puntos Global")
    print("="*50 + "\n")
    
    print("📷 Cargando YOLO...")
    detector = YOLO("yolo11n.pt")
    
    depth_estimator = None if args.no_depth else GlobalDepthEstimator()
    
    print(f"\n📹 Abriendo fuente: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la fuente")
        return
    
    print("✅ Listo! Presiona 'q' para salir")
    print(f"🎯 Detectando: Personas y Bicicletas")
    print(f"🔥 Mapa de calor: {'Activado' if args.show_heatmap else 'Desactivado'}\n")
    
    fps_buffer = deque(maxlen=30)
    frame_count = 0
    depth_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_total = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ============================================
        # 1. DETECCIÓN (YOLO)
        # ============================================
        start_det = time.time()
        results = detector(frame_rgb, verbose=False)
        det_time = (time.time() - start_det) * 1000
        
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if conf < CONFIDENCE_THRESHOLD:
                continue
            if cls not in CLASSES_INTERES:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': cls,
                'conf': conf,
                'distance': 0,
            })
        
        # ============================================
        # 2. PROFUNDIDAD GLOBAL (UNA VEZ POR FRAME)
        # ============================================
        depth_map = None
        depth_time = 0
        
        if depth_estimator and detections:
            start_depth = time.time()
            depth_map = depth_estimator.estimate_depth(frame_rgb)
            depth_time = (time.time() - start_depth) * 1000
            depth_times.append(depth_time)
            
            for det in detections:
                det['distance'] = depth_estimator.get_distance_from_bbox(
                    depth_map, det['bbox']
                )
        
        # ============================================
        # 3. VISUALIZACIÓN
        # ============================================
        start_viz = time.time()
        
        # Frame base
        frame_viz = frame.copy()
        
        # Si está activado el mapa de calor
        if args.show_heatmap and depth_map is not None:
            overlay, heatmap = GlobalDepthEstimator.create_heatmap_overlay(
                frame, depth_map, alpha=args.alpha
            )
            frame_viz = overlay
            
            # Mostrar barra de color (referencia)
            # Crear una barra de color vertical en el lateral derecho
            bar_height = 200
            bar_width = 30
            bar_x = frame_viz.shape[1] - bar_width - 10
            bar_y = 50
            
            # Generar gradiente de colores
            color_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
            for i in range(bar_height):
                value = int(255 * (1 - i / bar_height))
                color = cv2.applyColorMap(np.array([value], dtype=np.uint8).reshape(1, 1), cv2.COLORMAP_PLASMA)
                color_bar[i, :] = color[0, 0]
            
            # Pegar barra
            frame_viz[bar_y:bar_y+bar_height, bar_x:bar_x+bar_width] = color_bar
            
            # Etiquetas de distancia
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_viz, f"{MAX_DEPTH:.1f}m", 
                       (bar_x - 10, bar_y + 15), font, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_viz, "0m", 
                       (bar_x - 10, bar_y + bar_height + 15), font, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_viz, "Profundidad", 
                       (bar_x - 15, bar_y - 10), font, 0.4, (255, 255, 255), 1)
        
        # Dibujar detecciones
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['conf']
            distance = det.get('distance', 0)
            
            color = COLORS.get(cls, (255, 255, 255))
            label = LABELS.get(cls, f"Clase {cls}")
            
            # Bounding box
            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta
            cv2.putText(frame_viz, f"{label} {conf:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)
            
            # Distancia
            if distance > 0:
                dist_text = f"📏 {distance:.2f}m"
                cv2.putText(frame_viz, dist_text,
                           (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 255), 2)
                
                # Barra de distancia
                bar_width_dist = min(int(distance * 25), (x2 - x1))
                cv2.rectangle(frame_viz, (x1, y2 + 35), 
                             (x1 + bar_width_dist, y2 + 45), color, -1)
        
        viz_time = (time.time() - start_viz) * 1000
        
        # ============================================
        # 4. MÉTRICAS EN PANTALLA
        # ============================================
        total_time = (time.time() - start_total) * 1000
        fps_buffer.append(1000 / total_time if total_time > 0 else 0)
        frame_count += 1
        avg_fps = np.mean(fps_buffer) if fps_buffer else 0
        
        info_y = 30
        cv2.putText(frame_viz, f"FPS: {avg_fps:.1f}",
                   (frame_viz.shape[1] - 180, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame_viz, f"Dets: {len(detections)}",
                   (frame_viz.shape[1] - 180, info_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if depth_estimator:
            depth_status = f"Depth: {depth_time:.0f}ms" if depth_time > 0 else "Depth: ON"
            cv2.putText(frame_viz, depth_status,
                       (frame_viz.shape[1] - 180, info_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Estado del mapa de calor
        if args.show_heatmap:
            cv2.putText(frame_viz, "🔥 HEATMAP ON",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2)
        
        cv2.putText(frame_viz, f"Det: {det_time:.0f}ms  Viz: {viz_time:.0f}ms",
                   (10, frame_viz.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Mostrar
        cv2.imshow("SpeedrEye - Nube Global con Mapa de Calor", frame_viz)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ============================================
    # LIMPIEZA
    # ============================================
    cap.release()
    cv2.destroyAllWindows()
    
    avg_depth_time = np.mean(depth_times) if depth_times else 0
    
    print(f"\n✅ Procesamiento finalizado")
    print(f"   FPS promedio: {avg_fps:.1f}")
    print(f"   Tiempo depth: {avg_depth_time:.1f}ms" if avg_depth_time > 0 else "   Depth: OFF")
    print(f"   Total frames: {frame_count}")

if __name__ == "__main__":
    main()