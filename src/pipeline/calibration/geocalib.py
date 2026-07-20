"""
Módulo de calibración con GeoCalib
"""

import cv2
import numpy as np
import torch
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque

# Silenciar warnings
warnings.filterwarnings("ignore")

class CameraCalibrator:
    def __init__(self, config):
        self.config = config
        self.focal_length = config.FOCAL_LENGTH
        self.cx = config.CX
        self.cy = config.CY
        self.camera_matrix = None
        self.dist_coeffs = None
        self.is_calibrated = False
        self.calibration_info = {}
        self.geocalib = None
        self.vanishing_point = None
        self.horizon_line = None
        self.horizon_buffer = deque(maxlen=10)
        self.vanishing_buffer = deque(maxlen=10)
        
        print("📷 Inicializando Calibrador...")
        self._load_geocalib()
    
    def _load_geocalib(self):
        try:
            try:
                from geocalib import GeoCalib
                self.geocalib = GeoCalib
                return
            except ImportError:
                import subprocess, sys
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "git+https://github.com/cvg/GeoCalib.git"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                from geocalib import GeoCalib
                self.geocalib = GeoCalib
        except Exception:
            self.geocalib = None
    
    def calibrate_from_video(self, video_path, num_frames=30):
        print(f"\n🔧 Calibrando desde: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("   ❌ No se pudo abrir el video")
            return self._get_default_params()
        
        frames = []
        frame_count = 0
        h, w = 0, 0
        
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            h, w = frame.shape[:2]
            frame_count += 1
        
        cap.release()
        
        if not frames:
            print("   ❌ No se pudieron leer frames")
            return self._get_default_params()
        
        mid_idx = len(frames) // 2
        calib_frame = frames[mid_idx]
        video_name = Path(video_path).stem
        
        result = self._calibrate_with_geocalib(calib_frame)
        
        if result is None or self.focal_length < 300 or self.focal_length > 2000:
            self.focal_length = 700.0
            self.cx = w / 2
            self.cy = h / 2
            self.camera_matrix = np.array([
                [self.focal_length, 0, self.cx],
                [0, self.focal_length, self.cy],
                [0, 0, 1]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((1, 5), dtype=np.float32)
            self.is_calibrated = True
            self.calibration_info['method'] = 'default'
            print(f"   ✅ Calibrado por defecto: f={self.focal_length:.0f}px")
        
        self._detect_horizon_and_vp(calib_frame)
        self._generate_visualization(calib_frame, video_name)
        
        return self.get_parameters()
    
    def _calibrate_with_geocalib(self, img):
        if self.geocalib is None:
            return None
        
        try:
            h, w = img.shape[:2]
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            model = self.geocalib()
            results = model.calibrate(img_tensor, camera_model="simple_radial")
            
            if isinstance(results, dict):
                focal_norm = results.get('focal', 0.5)
                pp = results.get('principal_point', [0.5, 0.5])
                dist = results.get('distortion', [0.0, 0.0])
            else:
                focal_norm = results.focal if hasattr(results, 'focal') else 0.5
                pp = results.principal_point if hasattr(results, 'principal_point') else [0.5, 0.5]
                dist = results.distortion if hasattr(results, 'distortion') else [0.0, 0.0]
            
            focal = focal_norm * max(w, h)
            
            if focal < 300 or focal > 2000:
                focal = 700.0
                self.cx = w / 2
                self.cy = h / 2
            else:
                self.focal_length = focal
                self.cx = pp[0] * w
                self.cy = pp[1] * h
            
            k1 = dist[0] if len(dist) > 0 else 0.0
            k2 = dist[1] if len(dist) > 1 else 0.0
            self.dist_coeffs = np.array([[k1, k2, 0, 0, 0]], dtype=np.float32)
            
            self.camera_matrix = np.array([
                [self.focal_length, 0, self.cx],
                [0, self.focal_length, self.cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            self.is_calibrated = True
            self.calibration_info.update({
                'method': 'geocalib',
                'focal_pixels': float(self.focal_length),
                'image_size': (w, h),
            })
            
            print(f"   ✅ Calibrado: f={self.focal_length:.0f}px")
            return self.get_parameters()
            
        except Exception:
            return None
    
    def _detect_horizon_and_vp(self, image):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) < 5:
            self.vanishing_point = np.array([w/2, h/2])
            self.horizon_line = h/2
            return
        
        line_segments = []
        for line in lines:
            if line.ndim == 1 and len(line) == 4:
                line_segments.append(line)
            elif line.ndim == 2 and line.shape[0] == 1 and line.shape[1] == 4:
                line_segments.append(line[0])
        
        if len(line_segments) < 5:
            self.vanishing_point = np.array([w/2, h/2])
            self.horizon_line = h/2
            return
        
        vp_candidates = []
        for i in range(len(line_segments)):
            for j in range(i+1, len(line_segments)):
                x1, y1, x2, y2 = line_segments[i]
                x3, y3, x4, y4 = line_segments[j]
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) > 1e-6:
                    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
                    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
                    if 0 < px < w and 0 < py < h * 1.5:
                        vp_candidates.append((px, py))
        
        if vp_candidates:
            vp_x = np.median([p[0] for p in vp_candidates])
            vp_y = np.median([p[1] for p in vp_candidates])
            self.vanishing_point = np.array([vp_x, vp_y])
            self.horizon_line = vp_y
            print(f"   📐 Horizonte: y={vp_y:.0f}")
        else:
            self.vanishing_point = np.array([w/2, h/2])
            self.horizon_line = h/2
    
    def update_scene_params(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) < 5:
            return
        
        line_segments = []
        for line in lines:
            if line.ndim == 1 and len(line) == 4:
                line_segments.append(line)
            elif line.ndim == 2 and line.shape[0] == 1 and line.shape[1] == 4:
                line_segments.append(line[0])
        
        if len(line_segments) < 5:
            return
        
        vp_candidates = []
        for i in range(len(line_segments)):
            for j in range(i+1, len(line_segments)):
                x1, y1, x2, y2 = line_segments[i]
                x3, y3, x4, y4 = line_segments[j]
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) > 1e-6:
                    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
                    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
                    if 0 < px < w and 0 < py < h * 1.5:
                        vp_candidates.append((px, py))
        
        if vp_candidates:
            vp_x = np.median([p[0] for p in vp_candidates])
            vp_y = np.median([p[1] for p in vp_candidates])
            new_vp = np.array([vp_x, vp_y])
            new_horizon = vp_y
            
            self.vanishing_buffer.append(new_vp)
            self.horizon_buffer.append(new_horizon)
            
            if len(self.vanishing_buffer) > 5:
                vp_median = np.median(self.vanishing_buffer, axis=0)
                horizon_median = np.median(self.horizon_buffer)
                
                if self.vanishing_point is not None:
                    old_vp = self.vanishing_point
                    change = np.sqrt((vp_median[0] - old_vp[0])**2 + (vp_median[1] - old_vp[1])**2)
                    if change > h * 0.05:
                        print(f"   🔄 Cambio horizonte: y={self.horizon_line:.0f}→{horizon_median:.0f}")
                
                self.vanishing_point = vp_median
                self.horizon_line = horizon_median
    
    def _get_default_params(self):
        self.is_calibrated = False
        return self.get_parameters()
    
    def _generate_visualization(self, img, video_name):
        try:
            h, w = img.shape[:2]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            
            ax1 = axes[0]
            img_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
            
            if self.vanishing_point is not None:
                vx, vy = int(self.vanishing_point[0]), int(self.vanishing_point[1])
                cv2.circle(img_viz, (vx, vy), 12, (255, 0, 0), 3)
                cv2.putText(img_viz, "VP", (vx+15, vy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            if self.horizon_line is not None:
                hy = int(self.horizon_line)
                cv2.line(img_viz, (0, hy), (w, hy), (0, 255, 255), 1)
                cv2.putText(img_viz, "Horizonte", (10, hy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            cx, cy = int(self.cx), int(self.cy)
            cv2.circle(img_viz, (cx, cy), 6, (0, 255, 0), 2)
            cv2.putText(img_viz, "C", (cx+8, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            ax1.imshow(img_viz)
            ax1.set_title('Calibración', fontsize=10)
            ax1.axis('off')
            
            ax2 = axes[1]
            ax2.axis('off')
            
            info_text = f"""
            CALIBRACIÓN
            
            Focal: {self.focal_length:.0f} px
            Centro: ({self.cx:.0f}, {self.cy:.0f})
            Horizonte: y={self.horizon_line:.0f}
            """
            
            if self.vanishing_point is not None:
                info_text += f"\n   VP: ({self.vanishing_point[0]:.0f}, {self.vanishing_point[1]:.0f})"
            
            ax2.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace', transform=ax2.transAxes)
            ax2.set_title('Parámetros', fontsize=10)
            
            plt.tight_layout()
            
            # Guardar en results/calibration
            output_path = self.config.CALIBRATION_DIR / f"calibration_report_{video_name}.png"
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception:
            pass
    
    def undistort_frame(self, frame):
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = frame.shape[:2]
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h)
            )
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        return frame
    
    def get_parameters(self):
        return {
            'focal_length': self.focal_length,
            'cx': self.cx,
            'cy': self.cy,
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'is_calibrated': self.is_calibrated,
            'vanishing_point': self.vanishing_point,
            'horizon_y': self.horizon_line,
        }