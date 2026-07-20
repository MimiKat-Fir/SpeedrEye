"""
Módulo de estimadores de profundidad
"""

from .base import BaseDepthEstimator
from .midas import MidasDepthEstimator

def get_depth_estimator(config, engine=None):
    """
    Fábrica de estimadores de profundidad.
    Devuelve el estimador según la configuración.
    """
    if engine is None:
        engine = config.DEPTH_ENGINE
    
    engine = engine.lower()
    
    if engine == "midas":
        return MidasDepthEstimator(config)
    else:
        print(f"⚠️ Motor '{engine}' no reconocido. Usando MiDaS por defecto.")
        return MidasDepthEstimator(config)