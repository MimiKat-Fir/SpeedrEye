"""
Módulo de métricas de rendimiento
"""

from collections import deque
import numpy as np

class MetricsTracker:
    """Tracker de métricas de rendimiento"""
    
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.fps_buffer = deque(maxlen=buffer_size)
        self.frame_count = 0
        self.total_time = 0.0
        self.depth_times = []
        self.det_times = []
        self.viz_times = []
        
        # Últimas métricas
        self.last_metrics = {
            'fps': 0.0,
            'total_time': 0.0,
            'depth_time': 0.0,
            'det_time': 0.0,
            'viz_time': 0.0,
            'frame_count': 0,
            'avg_depth_time': 0.0,
        }
    
    def update(self, total_ms, det_ms, depth_ms, viz_ms):
        """Actualiza métricas con un nuevo frame"""
        self.frame_count += 1
        self.total_time += total_ms
        self.depth_times.append(depth_ms)
        self.det_times.append(det_ms)
        self.viz_times.append(viz_ms)
        
        fps = 1000 / total_ms if total_ms > 0 else 0
        self.fps_buffer.append(fps)
        
        self.last_metrics = {
            'fps': np.mean(self.fps_buffer) if self.fps_buffer else 0,
            'total_time': total_ms,
            'depth_time': depth_ms,
            'det_time': det_ms,
            'viz_time': viz_ms,
            'frame_count': self.frame_count,
            'avg_depth_time': np.mean(self.depth_times) if self.depth_times else 0,
            'avg_fps': self.frame_count / (self.total_time / 1000) if self.total_time > 0 else 0,
        }
        
        return self.last_metrics
    
    def get_summary(self):
        """Obtiene resumen final"""
        return {
            'fps_avg': self.frame_count / (self.total_time / 1000) if self.total_time > 0 else 0,
            'depth_avg': np.mean(self.depth_times) if self.depth_times else 0,
            'det_avg': np.mean(self.det_times) if self.det_times else 0,
            'viz_avg': np.mean(self.viz_times) if self.viz_times else 0,
            'total_frames': self.frame_count,
        }