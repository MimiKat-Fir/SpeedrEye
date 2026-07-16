#!/usr/bin/env python3
"""
SpeedrEye Modular - Punto de entrada principal
"""

import cv2
import time
import argparse
from collections import deque
import numpy as np

# Importaciones ABSOLUTAS (funciona al ejecutar desde personal/)
from speedreye_modular.config import Config
from speedreye_modular.detector import Detector
from speedreye_modular.depth_estimator import DepthEstimator
from speedreye_modular.visualizer import Visualizer
from speedreye_modular.metrics import MetricsTracker

def main():
    parser = argparse.ArgumentParser(description="SpeedrEye Modular")
    parser.add_argument("--video", type=str, help="Ruta a video (o 0 para cámara)")
    parser.add_argument("--camera", action="store_true", help="Usar cámara web")
    parser.add_argument("--no_depth", action="store_true", help="Desactivar profundidad")
    parser.add_argument("--show_heatmap", action="store_true", help="Mostrar mapa de calor")
    parser.add_argument("--alpha", type=float, default=Config.HEATMAP_ALPHA, 
                        help="Transparencia del mapa de calor")
    
    args = parser.parse_args()
    
    # Fuente de video
    if args.camera:
        video_path = 0
    elif args.video:
        video_path = args.video
    else:
        video_path = 0
    
    # ============================================
    # INICIALIZAR MÓDULOS
    # ============================================
    
    print("\n" + "="*50)
    print("🚗 SPEEDREYE MODULAR")
    print("="*50 + "\n")
    
    # Detector
    print("📷 Inicializando Detector...")
    detector = Detector(
        model_name=Config.YOLO_MODEL,
        confidence=Config.CONFIDENCE_THRESHOLD,
        classes=Config.CLASSES_INTERES
    )
    
    # Depth Estimator
    depth_estimator = None if args.no_depth else DepthEstimator(
        model_type=Config.DEPTH_MODEL,
        max_size=Config.MAX_SIZE,
        max_depth=Config.MAX_DEPTH,
        margin=Config.MARGIN,
        ground_ratio=Config.GROUND_RATIO
    )
    
    # Visualizer
    visualizer = Visualizer(Config)
    
    # Metrics
    metrics = MetricsTracker()
    
    print("\n📹 Abriendo fuente...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la fuente")
        return
    
    print("✅ Listo! Presiona 'q' para salir\n")
    
    # Buffer para suavizar distancias
    distance_buffer = {}
    
    # ============================================
    # BUCLE PRINCIPAL
    # ============================================
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_total = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Detección
        start_det = time.time()
        detections = detector.detect(frame_rgb)
        det_time = (time.time() - start_det) * 1000
        
        # 2. Profundidad (solo si hay detecciones y depth activado)
        depth_map = None
        depth_time = 0
        
        if depth_estimator and detections:
            start_depth = time.time()
            depth_map = depth_estimator.estimate_depth(frame_rgb)
            depth_time = (time.time() - start_depth) * 1000
            
            # Suavizar distancias con promedio móvil
            for det in detections:
                bbox = det['bbox']
                key = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                
                if key not in distance_buffer:
                    distance_buffer[key] = deque(maxlen=5)
                
                raw_distance = depth_estimator.get_distance_from_bbox(
                    depth_map, det['bbox']
                )
                distance_buffer[key].append(raw_distance)
                det['distance'] = np.mean(distance_buffer[key])
        
        # 3. Visualización
        start_viz = time.time()
        viz_frame = visualizer.draw(
            frame, detections, depth_map,
            show_heatmap=args.show_heatmap,
            show_distance=True,
            show_ui=True,
            metrics=metrics.last_metrics
        )
        viz_time = (time.time() - start_viz) * 1000
        
        # 4. Actualizar métricas
        total_time = (time.time() - start_total) * 1000
        metrics.update(total_time, det_time, depth_time, viz_time)
        
        # 5. Mostrar
        cv2.imshow("SpeedrEye - Modular", viz_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ============================================
    # FINALIZAR
    # ============================================
    
    cap.release()
    cv2.destroyAllWindows()
    
    summary = metrics.get_summary()
    print("\n✅ Procesamiento finalizado")
    print(f"   FPS promedio: {summary['fps_avg']:.1f}")
    print(f"   Tiempo depth: {summary['depth_avg']:.1f}ms")
    print(f"   Total frames: {summary['total_frames']}")

if __name__ == "__main__":
    main()