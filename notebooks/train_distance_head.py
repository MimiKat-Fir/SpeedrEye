#!/usr/bin/env python3
"""
Paso 2 del pipeline SpeedrEye: entrena la cabecita de corrección de distancia
("geometry_guided.pt"). NO es un YOLO. Es una MLP diminuta.

Idea:
    distancia_final = distancia_geometrica * factor_aprendido

- distancia_geometrica sale de una fórmula fija (altura_real_clase * focal / altura_bbox_px),
  igual que en tu geometry_guided.py de producción.
- distancia_real sale de las etiquetas 3D de KITTI (campo 'location', eje Z),
  que fueron construidas por KITTI usando LIDAR. Aquí el LIDAR se usa
  SOLO para tener ese número de referencia durante el entrenamiento;
  el modelo entrenado nunca necesita LIDAR en producción.
- El detector YOLO se usa CONGELADO, solo para extraer su feature map;
  no se actualiza ningún peso del detector aquí.

Requiere que el detector (Paso 1) ya esté entrenado, porque la cabecita
aprende sobre las features que ese detector concreto produce.

Uso:
    python train_distance_head.py \
        --kitti-root /ruta/a/KITTI/object_detection \
        --detector-weights ./kitti_yolo/speedreye_detector/weights/best.pt \
        --output ./models/distance/geometry_guided.pt \
        --epochs 40
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Reutilizamos la implementación ya existente en el pipeline: la ROI-align,
# la captura de features, la cabecita y el guardado de checkpoints deben
# ser IDÉNTICOS a los que usa main.py en producción, para que el checkpoint
# resultante sea compatible con load_distance_head().
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline.distance.head import (  # noqa: E402
    BackboneFeatureCapture,
    DistanceRegressionHead,
    prepare_distance_inputs,
    file_sha256,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_download import ensure_kitti_downloaded, DEFAULT_KITTI_DIR, PROJECT_ROOT  # noqa: E402

# Debe coincidir con Config.CLASS_HEIGHTS del pipeline de producción.
# Es A PROPÓSITO una constante por clase (no la altura real de cada KITTI
# GT): en inferencia no conocemos la altura real de cada persona, así que
# entrenamos con la misma aproximación que se usará después. La cabecita
# aprende a corregir precisamente ESA discrepancia (postura, oclusión,
# variación de altura real, ángulo de cámara), no a hacer trampa con datos
# que no tendrá disponibles en producción.
CLASS_HEIGHTS = {0: 1.70, 1: 1.70}
KITTI_TO_YOLO_CLASS = {"Pedestrian": 0, "Cyclist": 1}
MIN_DISTANCE, MAX_DISTANCE = 0.5, 60.0


def parse_calib(calib_path):
    """Extrae fx, fy, cx, cy de la matriz P2 (cámara color izquierda) de KITTI."""
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                values = list(map(float, line.strip().split(" ")[1:]))
                p2 = np.array(values).reshape(3, 4)
                return {
                    "fx": p2[0, 0], "fy": p2[1, 1],
                    "cx": p2[0, 2], "cy": p2[1, 2],
                }
    raise ValueError(f"No se encontró P2 en {calib_path}")


def parse_kitti_full_label(label_path):
    """Lee label_2 con bbox 2D + posición 3D (location), para Pedestrian/Cyclist."""
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 15:
                continue
            obj_type = parts[0]
            if obj_type not in KITTI_TO_YOLO_CLASS:
                continue
            bbox = tuple(map(float, parts[4:8]))          # x1,y1,x2,y2
            location = tuple(map(float, parts[11:14]))    # x,y,z (coords cámara)
            objects.append({
                "class": KITTI_TO_YOLO_CLASS[obj_type],
                "bbox": bbox,
                "z_meters": location[2],  # profundidad real (eje óptico), viene del LIDAR
            })
    return objects


class KittiDistanceDataset(Dataset):
    """Un ítem = una imagen completa + sus objetos relevantes con GT de distancia."""

    def __init__(self, kitti_root: Path, file_ids):
        self.kitti_root = kitti_root
        self.file_ids = file_ids

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        img_path = self.kitti_root / "training" / "image_2" / f"{file_id}.png"
        label_path = self.kitti_root / "training" / "label_2" / f"{file_id}.txt"
        calib_path = self.kitti_root / "training" / "calib" / f"{file_id}.txt"

        image = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]  # RGB->BGR para cv2/ultralytics
        objects = parse_kitti_full_label(label_path)
        calib = parse_calib(calib_path)
        return image, objects, calib


def collate_single(batch):
    return batch[0]  # procesamos imagen a imagen (el nº de objetos por imagen varía)


def compute_targets(objects, calib, image_shape):
    """
    Para cada objeto: distancia geométrica (fórmula fija) vs distancia real
    (KITTI/LIDAR). Devuelve detections-like dicts + tensor de targets (log-factor).
    """
    dets, targets = [], []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        box_h_px = max(y2 - y1, 1.0)
        real_height = CLASS_HEIGHTS[obj["class"]]

        dist_geom = real_height * calib["fx"] / box_h_px
        dist_geom = min(max(dist_geom, MIN_DISTANCE), MAX_DISTANCE)

        dist_real = obj["z_meters"]
        if dist_real <= 0:
            continue  # etiqueta inválida, se descarta

        factor = dist_real / dist_geom
        factor = min(max(factor, 0.2), 5.0)  # recorte de outliers/errores de anotación

        dets.append({"bbox": (x1, y1, x2, y2), "class": obj["class"]})
        targets.append(np.log(factor))

    return dets, torch.tensor(targets, dtype=torch.float32)


def get_file_ids(kitti_root: Path, val_ratio: float, seed: int = 42):
    label_dir = kitti_root / "training" / "label_2"
    ids = sorted(p.stem for p in label_dir.glob("*.txt"))

    random.seed(seed)
    block_size = 10
    blocks = [ids[i:i + block_size] for i in range(0, len(ids), block_size)]
    random.shuffle(blocks)
    n_val = max(1, int(len(blocks) * val_ratio))
    val_ids = [i for b in blocks[:n_val] for i in b]
    train_ids = [i for b in blocks[n_val:] for i in b]
    return train_ids, val_ids


def run_epoch(loader, detector_model, capture, head, optimizer, device, train_mode: bool):
    head.train(train_mode)
    total_loss, n_batches = 0.0, 0

    for image, objects, calib in loader:
        if not objects:
            continue

        dets, targets = compute_targets(objects, calib, image.shape)
        if not dets:
            continue
        targets = targets.to(device)

        # Forward por el detector SOLO para disparar el hook y capturar
        # el feature map de esta imagen concreta. No se usan sus detecciones.
        with torch.no_grad():
            detector_model.predict(image, verbose=False, conf=0.001, imgsz=640)

        if not capture.features:
            continue
        feature_map = capture.features[0].to(device)

        rois, box_features = prepare_distance_inputs(dets, image.shape, feature_map, capture.strides[0])
        rois, box_features = rois.to(device), box_features.to(device)

        if train_mode:
            optimizer.zero_grad()
            preds = head(feature_map, rois, box_features)
            loss = nn.functional.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = head(feature_map, rois, box_features)
                loss = nn.functional.mse_loss(preds, targets)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Entrena la cabecita de corrección geometry_guided")
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_DIR,
                         help=f"Por defecto: {DEFAULT_KITTI_DIR} (se descarga solo si falta)")
    parser.add_argument("--detector-weights", type=Path, required=True,
                         help="Pesos ya entrenados por train_detector.py (best.pt)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "distance" / "geometry_guided_v1.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--roi-size", type=int, default=3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--no-download", action="store_true",
                         help="No intentar descargar KITTI aunque falte en --kitti-root")
    args = parser.parse_args()

    if not args.no_download:
        ensure_kitti_downloaded(args.kitti_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    from ultralytics import YOLO

    print("📦 Cargando detector congelado...")
    detector = YOLO(str(args.detector_weights))
    detector.model.to(device)
    for p in detector.model.parameters():
        p.requires_grad_(False)
    detector.model.eval()

    capture = BackboneFeatureCapture(detector.model.model)

    # Averigua el nº de canales del feature map con una pasada dummy.
    dummy = np.zeros((384, 1280, 3), dtype=np.uint8)
    with torch.no_grad():
        detector.predict(dummy, verbose=False, conf=0.001, imgsz=640)
    feature_channels = capture.features[0].shape[1]
    print(f"   Canales de feature map detectados: {feature_channels}")

    head = DistanceRegressionHead(
        feature_channels=feature_channels,
        hidden_channels=args.hidden_channels,
        roi_size=args.roi_size,
    ).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    train_ids, val_ids = get_file_ids(args.kitti_root, args.val_ratio)
    print(f"   Train: {len(train_ids)} imágenes | Val: {len(val_ids)} imágenes")

    train_loader = DataLoader(
        KittiDistanceDataset(args.kitti_root, train_ids),
        batch_size=1, shuffle=True, collate_fn=collate_single, num_workers=2,
    )
    val_loader = DataLoader(
        KittiDistanceDataset(args.kitti_root, val_ids),
        batch_size=1, shuffle=False, collate_fn=collate_single, num_workers=2,
    )

    best_val_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, detector, capture, head, optimizer, device, train_mode=True)
        val_loss = run_epoch(val_loader, detector, capture, head, optimizer, device, train_mode=False)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss(log-factor) {train_loss:.4f} | val_loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "target_mode": "geometry_guided",
                "feature_channels": feature_channels,
                "hidden_channels": args.hidden_channels,
                "roi_size": args.roi_size,
                "head_state_dict": head.state_dict(),
                "detector_weights": args.detector_weights.name,
                "detector_sha256": file_sha256(args.detector_weights),
            }
            torch.save(checkpoint, args.output)
            print(f"   💾 Nuevo mejor checkpoint guardado en {args.output} (val_loss={val_loss:.4f})")

    print(f"\n✅ Entrenamiento de la cabecita terminado. Mejor val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
