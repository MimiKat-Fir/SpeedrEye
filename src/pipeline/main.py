#!/usr/bin/env python3
"""
SpeedrEye Pipeline - Detección y Distancia en tiempo real
"""

import os
import sys
import warnings
import argparse
import time
from pathlib import Path
from collections import deque
import numpy as np

# 🔥 SILENCIAR TODO
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ["QT_QPA_PLATFORM"] = ""
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Silenciar logs de PyTorch y ultralytics
import torch
torch.set_printoptions(precision=2, sci_mode=False)

import cv2
import logging

# Silenciar logs de ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

cv2.ocl.setUseOpenCL(False)

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import Config
from pipeline.detector import Detector
from pipeline.depth_estimators import get_depth_estimator
from pipeline.smart_lidar import SmartLidarGenerator
from pipeline.fusion import Fusion
from pipeline.visualizer import Visualizer
from pipeline.calibration import CameraCalibrator

class SpeedrEyePipeline:
    def __init__(self, config, video_path=None):
        self.config = config
        self.fps_buffer = deque(maxlen=30)
        self.frame_count = 0
        
        print("\n" + "="*50)
        print("SPEEDREYE PIPELINE")
        print("="*50 + "\n")
        
        self.calibrator = CameraCalibrator(config)
        if video_path and Path(video_path).exists():
            self.calibrator.calibrate_from_video(video_path, num_frames=20)
        
        params = self.calibrator.get_parameters()
        config.FOCAL_LENGTH = params['focal_length']
        config.CX = params['cx']
        config.CY = params['cy']
        
        print("\nInicializando...")
        self.detector = Detector(config)
        self.depth_estimator = get_depth_estimator(config)
        self.smart_lidar = SmartLidarGenerator(config)
        self.fusion = Fusion(config)
        self.visualizer = Visualizer(config)
        
        print("✅ Listo!\n")
    
    def process_frame(self, frame, show_heatmap=False, show_clouds=False):
        start_total = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detections = self.detector.detect(frame_rgb)
        det_time = (time.time() - start_total) * 1000
        
        depth_map = None
        depth_time = 0
        fusion_time = 0
        
        if detections:
            start_depth = time.time()
            depth_map = self.depth_estimator.estimate_depth(frame_rgb)
            depth_time = (time.time() - start_depth) * 1000
            
            start_fusion = time.time()
            detections = self.fusion.fuse(detections, depth_map, frame_rgb)
            fusion_time = (time.time() - start_fusion) * 1000
        
        if self.frame_count % 10 == 0 and self.calibrator.is_calibrated:
            self.calibrator.update_scene_params(frame)
        
        # Filtrar por horizonte
        if self.calibrator.horizon_line is not None:
            horizon_y = self.calibrator.horizon_line
            filtered = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                center_y = (y1 + y2) / 2
                if center_y > horizon_y:
                    filtered.append(det)
            detections = filtered
        
        metrics = {
            'fps': np.mean(self.fps_buffer) if self.fps_buffer else 0,
            'detections': len(detections),
            'depth_time': depth_time,
            'fusion_time': fusion_time,
            'total_time': (time.time() - start_total) * 1000,
            'focal_length': self.config.FOCAL_LENGTH,
        }
        
        viz_frame = self.visualizer.draw(
            frame, detections, depth_map,
            show_heatmap=show_heatmap,
            show_distance=True,
            show_clouds=show_clouds,
            metrics=metrics
        )
        
        self.fps_buffer.append(1000 / metrics['total_time'] if metrics['total_time'] > 0 else 0)
        self.frame_count += 1
        
        return viz_frame, detections, metrics
    
    def run_video(self, video_path, show_heatmap=False, show_clouds=False):
        if isinstance(video_path, str) and not Path(video_path).exists():
            print(f"❌ No se encuentra: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ No se pudo abrir la fuente")
            return
        
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1.0 / original_fps
        print(f"📹 FPS video: {original_fps:.1f} | Presiona 'q' para salir\n")
        
        last_frame_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            if current_time - last_frame_time < frame_time:
                time.sleep(frame_time - (current_time - last_frame_time))
            
            viz_frame, detections, metrics = self.process_frame(frame, show_heatmap, show_clouds)
            last_frame_time = time.time()
            
            cv2.putText(viz_frame, f"FPS: {metrics['fps']:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(viz_frame, f"Dets: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(viz_frame, f"Focal: {metrics['focal_length']:.0f}px", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.imshow("SpeedrEye", viz_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Finalizado | FPS: {np.mean(self.fps_buffer):.1f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Ruta al video")
    parser.add_argument("--camera", action="store_true", help="Usar cámara")
    parser.add_argument("--show_heatmap", action="store_true")
    parser.add_argument("--show_clouds", action="store_true")
    parser.add_argument("--depth_engine", type=str, default="midas",
                        choices=["midas", "depth_anything"])
    
    args = parser.parse_args()
    
    video_path = 0 if args.camera else args.video
    if not video_path and not args.camera:
        video_path = 0
    
    Config.DEPTH_ENGINE = args.depth_engine
    
    pipeline = SpeedrEyePipeline(Config, video_path=video_path)
    pipeline.run_video(video_path, show_heatmap=args.show_heatmap, 
                       show_clouds=args.show_clouds)

if __name__ == "__main__":
    main()