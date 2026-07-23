"""SpeedrEye detection pipeline."""

from .detector import Detector
from .distance import build_distance_estimator
from .visualizer import Visualizer
from .config import Config
from .calibration import CameraCalibrator


#cambios
def __init__(self, config, video_path=None, distance_method=None):
        self.config = config
        self.fps_buffer = deque(maxlen=config.FPS_BUFFER_SIZE)
        self.frame_count = 0
        
        # --- NUEVO: Memoria de seguimiento de Filtros de Kalman ---
        self.trackers = {}  # Formato: { track_id: KalmanPredictor() }
        # -----------------------------------------------------------

        self.calibrator = CameraCalibrator(config)
        # ... resto del init igual ...