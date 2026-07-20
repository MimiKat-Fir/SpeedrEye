"""
SpeedrEye Pipeline - Detección, profundidad, Pseudo-LiDAR y predicción
"""

from .detector import Detector
from .depth_estimators import get_depth_estimator
from .smart_lidar import SmartLidarGenerator
from .fusion import Fusion
from .visualizer import Visualizer
from .config import Config
from .calibration import CameraCalibrator