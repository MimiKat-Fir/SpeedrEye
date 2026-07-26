"""
Módulo de calibración de cámara (GeoCalib + detección de horizonte).
"""

import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")


class CameraCalibrator:
    def __init__(self, config):
        self.config = config
        self.focal_length = config.FOCAL_LENGTH
        self.cx = config.CX
        self.cy = config.CY
        self.is_calibrated = False
        self.calibration_method = None
        self.geocalib = None
        self.vanishing_point = None
        self.horizon_line = None
        self.horizon_buffer = deque(maxlen=10)
        self.vanishing_buffer = deque(maxlen=10)

        print("📷 Inicializando Calibrador...")
        self._load_geocalib()

    def _load_geocalib(self):
        """
        Carga GeoCalib si está instalado. Ya NO instala nada automáticamente:
        si falta, avisa con claridad y el calibrador sigue funcionando con
        una focal por defecto (Config.FOCAL_LENGTH). Instalar en silencio un
        paquete desde git en tiempo de ejecución era frágil (falla en redes
        restringidas, oculta errores) e innecesario si ya está en tu venv.
        """
        try:
            from geocalib import GeoCalib
            self.geocalib = GeoCalib
        except ImportError:
            print("   ⚠️  Paquete 'geocalib' no instalado. Para calibración automática:")
            print("       pip install git+https://github.com/cvg/GeoCalib.git")
            print("   ↪️  Usando calibración por defecto (focal fija) mientras tanto.")
            self.geocalib = None

    def calibrate_from_video(self, video_path, num_frames=20):
        print(f"\n🔧 Calibrando desde: {Path(video_path).name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("   ❌ No se pudo abrir el video")
            return self._get_default_params()

        # Antes se leían hasta `num_frames` frames completos a una lista en
        # RAM solo para quedarse con el del medio. Basta con saltar
        # directamente a ese frame.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_idx = min(num_frames // 2, max(total_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, calib_frame = cap.read()
        cap.release()

        if not ret or calib_frame is None:
            print("   ❌ No se pudo leer un frame de calibración")
            return self._get_default_params()

        h, w = calib_frame.shape[:2]
        video_name = Path(video_path).stem

        if not self._calibrate_with_geocalib(calib_frame):
            self._set_default_intrinsics(w, h)

        self._estimate_horizon(calib_frame, is_initial=True)

        if getattr(self.config, "SAVE_CALIBRATION_REPORT", True):
            self._generate_visualization(calib_frame, video_name)

        return self.get_parameters()

    def _set_default_intrinsics(self, w, h):
        self.focal_length = 700.0
        self.cx = w / 2
        self.cy = h / 2
        self.is_calibrated = True
        self.calibration_method = "default"
        print(f"   ✅ Calibrado por defecto: f={self.focal_length:.0f}px")

    def _calibrate_with_geocalib(self, img):
        """
        Intenta calibrar con GeoCalib.
        Devuelve True si obtuvo un resultado físicamente plausible
        (focal entre 300 y 2000px), False si hay que usar el fallback.
        """
        if self.geocalib is None:
            return False

        try:
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            model = self.geocalib()
            results = model.calibrate(img_tensor, camera_model="simple_radial")

            if isinstance(results, dict):
                focal_norm = results.get("focal", 0.5)
                pp = results.get("principal_point", [0.5, 0.5])
            else:
                focal_norm = getattr(results, "focal", 0.5)
                pp = getattr(results, "principal_point", [0.5, 0.5])

            focal = focal_norm * max(w, h)
            if not (300 <= focal <= 2000):
                return False  # fuera de rango plausible -> que decida el fallback

            self.focal_length = focal
            self.cx = pp[0] * w
            self.cy = pp[1] * h
            self.is_calibrated = True
            self.calibration_method = "geocalib"
            print(f"   ✅ Calibrado: f={self.focal_length:.0f}px")
            return True

        except Exception:
            return False

    # ------------------------------------------------------------------
    # Horizonte / punto de fuga
    # ------------------------------------------------------------------
    def _find_vanishing_point(self, frame, max_lines=40, min_angle_diff=0.15):
        """
        Detecta el punto de fuga dominante a partir de líneas de Hough.

        Dos mejoras respecto a la versión anterior:
        - Solo se consideran las `max_lines` líneas más largas (más
          fiables, y acota el coste O(n²) de las intersecciones por pares
          en vez de crecer con TODAS las líneas detectadas).
        - Se descartan pares de líneas casi paralelas (`min_angle_diff` en
          radianes): sus intersecciones son numéricamente inestables y son
          la principal fuente de outliers que antes disparaban el
          horizonte a valores absurdos.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) < 5:
            return None

        segments = [line[0] for line in lines]
        segments.sort(key=lambda s: np.hypot(s[2] - s[0], s[3] - s[1]), reverse=True)
        segments = segments[:max_lines]

        candidates = []
        for i in range(len(segments)):
            x1, y1, x2, y2 = segments[i]
            angle_i = np.arctan2(y2 - y1, x2 - x1)
            for j in range(i + 1, len(segments)):
                x3, y3, x4, y4 = segments[j]
                angle_j = np.arctan2(y4 - y3, x4 - x3)

                if abs(np.sin(angle_i - angle_j)) < min_angle_diff:
                    continue  # casi paralelas, se descarta el par

                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-6:
                    continue

                px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
                if 0 < px < w and 0 < py < h * 1.5:
                    candidates.append((px, py))

        if not candidates:
            return None

        vp_x = np.median([c[0] for c in candidates])
        vp_y = np.median([c[1] for c in candidates])
        return np.array([vp_x, vp_y])

    def _estimate_horizon(self, frame, is_initial=False):
        vp = self._find_vanishing_point(frame)
        h, w = frame.shape[:2]

        if vp is None:
            if is_initial:
                self.vanishing_point = np.array([w / 2, h / 2])
                self.horizon_line = h / 2
            return

        if is_initial:
            self.vanishing_point = vp
            self.horizon_line = vp[1]
            print(f"   📐 Horizonte: y={vp[1]:.0f}")
            return

        # Actualización incremental (solo se llega aquí si
        # Config.ENABLE_SCENE_RECALIBRATION=True, ver main.py)
        self.vanishing_buffer.append(vp)
        self.horizon_buffer.append(vp[1])
        if len(self.vanishing_buffer) <= 5:
            return

        vp_median = np.median(self.vanishing_buffer, axis=0)
        horizon_median = np.median(self.horizon_buffer)

        max_jump = h * self.config.HORIZON_MAX_JUMP_RATIO
        if self.vanishing_point is not None:
            change = float(np.linalg.norm(vp_median - self.vanishing_point))
            if change > max_jump:
                return  # salto implausible, probablemente ruido de Hough

        self.vanishing_point = vp_median
        self.horizon_line = horizon_median

    def update_scene_params(self, frame):
        """Recalibración incremental del horizonte (opcional, ver Config.ENABLE_SCENE_RECALIBRATION)."""
        self._estimate_horizon(frame, is_initial=False)

    def _get_default_params(self):
        self.is_calibrated = False
        return self.get_parameters()

    def _generate_visualization(self, img, video_name):
        try:
            import matplotlib.pyplot as plt  # import perezoso: solo si se pide el informe

            h, w = img.shape[:2]
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

            ax1 = axes[0]
            img_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

            if self.vanishing_point is not None:
                vx, vy = int(self.vanishing_point[0]), int(self.vanishing_point[1])
                cv2.circle(img_viz, (vx, vy), 12, (255, 0, 0), 3)
                cv2.putText(img_viz, "VP", (vx + 15, vy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            if self.horizon_line is not None:
                hy = int(self.horizon_line)
                cv2.line(img_viz, (0, hy), (w, hy), (0, 255, 255), 1)
                cv2.putText(img_viz, "Horizonte", (10, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cx, cy = int(self.cx), int(self.cy)
            cv2.circle(img_viz, (cx, cy), 6, (0, 255, 0), 2)
            cv2.putText(img_viz, "C", (cx + 8, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            ax1.imshow(img_viz)
            ax1.set_title("Calibración", fontsize=10)
            ax1.axis("off")

            ax2 = axes[1]
            ax2.axis("off")
            info_text = (
                f"CALIBRACIÓN\n\n"
                f"Método: {self.calibration_method}\n"
                f"Focal: {self.focal_length:.0f} px\n"
                f"Centro: ({self.cx:.0f}, {self.cy:.0f})\n"
                f"Horizonte: y={self.horizon_line:.0f}"
            )
            if self.vanishing_point is not None:
                info_text += f"\nVP: ({self.vanishing_point[0]:.0f}, {self.vanishing_point[1]:.0f})"

            ax2.text(0.1, 0.5, info_text, fontsize=10, verticalalignment="center",
                      fontfamily="monospace", transform=ax2.transAxes)
            ax2.set_title("Parámetros", fontsize=10)

            plt.tight_layout()
            output_path = self.config.CALIBRATION_DIR / f"calibration_report_{video_name}.png"
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

    def get_parameters(self):
        return {
            "focal_length": self.focal_length,
            "cx": self.cx,
            "cy": self.cy,
            "is_calibrated": self.is_calibrated,
            "calibration_method": self.calibration_method,
            "vanishing_point": self.vanishing_point,
            "horizon_y": self.horizon_line,
        }