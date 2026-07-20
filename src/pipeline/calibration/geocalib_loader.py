# pipeline/calibration/geocalib_loader.py

from geocalib import GeoCalib
import cv2
import numpy as np

class GeoCalibLoader:
    def __init__(self, model_type="pinhole"): 
        # "pinhole" para imágenes sin distorsión o con distorsión débil
        # "distorted" para imágenes con distorsión significativa
        self.model = GeoCalib(weights=model_type)

    def calibrate_from_image(self, image_path):
        # Carga la imagen
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ejecuta la calibración
        # Puedes elegir el modelo de cámara: "pinhole", "simple_radial", etc.
        result = self.model.calibrate(img_rgb, camera_model="simple_radial")

        # Extrae los parámetros: result.focal, result.principal_point, result.distortion
        return result