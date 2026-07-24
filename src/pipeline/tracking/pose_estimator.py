import numpy as np
from ultralytics import YOLO

class PoseOrientationEstimator:
    def __init__(self, model_path="yolov8n-pose.pt"):
        # Cargamos el modelo nano de pose (súper ligero, ~6MB)
        self.model = YOLO(model_path)

    def get_orientations(self, frame, detections):
        """
        Recibe el frame y la lista de detecciones de tu pipeline.
        Devuelve un diccionario {track_id: (dx, dy)} con el vector de dirección del cuerpo.
        """
        orientations = {}
        if not detections:
            return orientations

        # Ejecutamos pose en el frame
        results = self.model(frame, verbose=False, conf=0.3)[0]
        
        if results.keypoints is None or len(results.keypoints) == 0:
            return orientations

        # Convertimos keypoints a numpy: forma [N_personas, 17, 3] (x, y, conf)
        kpts_data = results.keypoints.data.cpu().numpy()
        boxes_data = results.boxes.xyxy.cpu().numpy()

        # Emparejamos cada detección de tu YOLO actual con la pose más cercana por lo que ocupan en pantalla
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            x1, y1, x2, y2 = det["bbox"]
            det_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            # Buscamos qué pose de YOLO-Pose coincide con la caja de tu YOLO principal
            best_idx = -1
            min_dist = float("inf")

            for idx, p_box in enumerate(boxes_data):
                p_center = np.array([(p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2])
                dist = np.linalg.norm(det_center - p_center)
                if dist < min_dist and dist < 50: # Umbral de tolerancia en píxeles
                    min_dist = dist
                    best_idx = idx

            if best_idx != -1:
                person_kpts = kpts_data[best_idx]
                dx, dy = self._calculate_vector_from_kpts(person_kpts)
                orientations[track_id] = (dx, dy)

        return orientations

    def _calculate_vector_from_kpts(self, kpts):
        """
        Calcula el vector de dirección teniendo en cuenta hombros, pies y fuga.
        """
        l_shoulder, r_shoulder = kpts[5], kpts[6]
        l_ankle, r_ankle = kpts[15], kpts[16] # Tobillos (índices 15 y 16)

        # 1. Intentamos medir la dirección de la zancada usando los pies/tobillos
        feet_dx = 0.0
        if l_ankle[2] > 0.3 and r_ankle[2] > 0.3:
            # Si un pie está más adelante que el otro en la pantalla
            feet_dx = (l_ankle[0] - r_ankle[0]) * 0.1

        # 2. Vector de hombros
        p1, p2 = None, None
        if l_shoulder[2] > 0.4 and r_shoulder[2] > 0.4:
            p1, p2 = l_shoulder[:2], r_shoulder[:2]

        if p1 is None or p2 is None:
            return 0.0, -1.0

        shoulder_vec = p2 - p1
        
        # Perpendicular al pecho
        normal_vec = np.array([-shoulder_vec[1], shoulder_vec[0]])
        norm = np.linalg.norm(normal_vec)
        
        if norm == 0:
            return 0.0, -1.0
            
        normal_vec = normal_vec / norm

        if normal_vec[1] > 0:
            normal_vec = -normal_vec

        # Si el vector es casi vertical (persona caminando de frente/espaldas),
        # añadimos el ajuste de zancada para que siga la curva natural del camino
        dx = float(normal_vec[0]) + feet_dx
        dy = float(normal_vec[1])

        # Normalizamos de nuevo
        total_norm = np.sqrt(dx**2 + dy**2)
        if total_norm > 0:
            dx /= total_norm
            dy /= total_norm

        return dx, dy