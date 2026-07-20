"""
Módulo de estimadores de profundidad
"""

from .base import BaseDepthEstimator
from .midas import MidasDepthEstimator
from .depth_anything import DepthAnythingEstimator

def get_depth_estimator(config, engine=None):
    """
    Fábrica de estimadores de profundidad.
    Devuelve el estimador según la configuración o el parámetro.
    """
    if engine is None:
        engine = config.DEPTH_ENGINE
    
    engine = engine.lower()
    
    if engine == "midas":
        return MidasDepthEstimator(config)
    elif engine == "depth_anything":
        return DepthAnythingEstimator(config)
    else:
        print(f"⚠️ Motor '{engine}' no reconocido. Usando MiDaS por defecto.")
        return MidasDepthEstimator(config)