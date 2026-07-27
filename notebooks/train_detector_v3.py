#!/usr/bin/env python3
"""
Paso 1 del pipeline SpeedrEye - VERSIÓN V3 OPTIMIZADA
Usa los datos KITTI ya descargados en data/kitti_raw/
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# RUTAS FIJAS (usando datos en data/ de la raíz)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KITTI_ROOT = PROJECT_ROOT / "data" / "kitti_raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "kitti_yolo_v3"
MODELS_DIR = PROJECT_ROOT / "models" / "yolo"

KITTI_TO_YOLO_CLASS = {
    "Pedestrian": 0,
    "Cyclist": 1,
}


def parse_kitti_label(label_path):
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 15:
                continue
            obj_type = parts[0]
            bbox = tuple(map(float, parts[4:8]))
            objects.append({"type": obj_type, "bbox": bbox})
    return objects


def create_balanced_dataset(kitti_root: Path, output_dir: Path, val_ratio: float = 0.15, seed: int = 42):
    """Crea dataset balanceado sobremuestreando ciclistas"""
    images_dir = kitti_root / "training" / "image_2"
    labels_dir = kitti_root / "training" / "label_2"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"No encuentro {images_dir} o {labels_dir}")

    image_files = sorted(images_dir.glob("*.png"))
    print(f"📦 {len(image_files)} imágenes encontradas en KITTI")

    # Clasificar imágenes por contenido
    images_with_cyclist = []
    images_with_pedestrian = []
    images_with_both = []

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        objects = parse_kitti_label(label_path)
        has_pedestrian = any(o["type"] == "Pedestrian" for o in objects)
        has_cyclist = any(o["type"] == "Cyclist" for o in objects)

        if has_pedestrian and has_cyclist:
            images_with_both.append(img_path)
        elif has_cyclist:
            images_with_cyclist.append(img_path)
        elif has_pedestrian:
            images_with_pedestrian.append(img_path)

    print(f"   🚶 Peatones: {len(images_with_pedestrian)}")
    print(f"   🚴 Ciclistas: {len(images_with_cyclist)}")
    print(f"   📸 Ambos: {len(images_with_both)}")

    # Sobremuestrear ciclistas 3x para balancear
    cyclist_oversampled = images_with_cyclist * 3
    both_oversampled = images_with_both * 2

    all_images = images_with_pedestrian + cyclist_oversampled + both_oversampled
    random.seed(seed)
    random.shuffle(all_images)

    print(f"   📊 Dataset balanceado: {len(all_images)} imágenes")

    # Split en bloques
    n = len(all_images)
    block_size = 10
    blocks = [all_images[i:i + block_size] for i in range(0, n, block_size)]
    random.shuffle(blocks)

    n_val_blocks = max(1, int(len(blocks) * val_ratio))
    val_blocks = blocks[:n_val_blocks]
    train_blocks = blocks[n_val_blocks:]

    train_files = [f for block in train_blocks for f in block]
    val_files = [f for block in val_blocks for f in block]

    print(f"   Train: {len(train_files)} | Val: {len(val_files)}")

    # Procesar y guardar
    kept_boxes = 0
    dropped_images = 0

    for split_name, files in (("train", train_files), ("val", val_files)):
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            objects = parse_kitti_label(label_path)
            relevant = [o for o in objects if o["type"] in KITTI_TO_YOLO_CLASS]

            if not relevant:
                dropped_images += 1
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            yolo_lines = []
            for obj in relevant:
                x1, y1, x2, y2 = obj["bbox"]
                x1, x2 = max(0, x1), min(img_w, x2)
                y1, y2 = max(0, y1), min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_id = KITTI_TO_YOLO_CLASS[obj["type"]]
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                kept_boxes += 1

            if not yolo_lines:
                dropped_images += 1
                continue

            shutil.copy(img_path, img_out / img_path.name)
            with open(lbl_out / f"{img_path.stem}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

    print(f"✅ Cajas conservadas: {kept_boxes}")
    print(f"⏭️  Imágenes descartadas: {dropped_images}")

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {output_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n"
            f"  0: Pedestrian\n"
            f"  1: Cyclist\n"
        )
    return yaml_path


def train_with_checkpoints(dataset_yaml: Path, model_name: str, imgsz: int, epochs: int,
                           output_dir: Path, models_dir: Path, output_name: str):
    from ultralytics import YOLO

    model = YOLO(model_name)

    model.train(
        data=str(dataset_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=0.9,  # 90% de VRAM
        
        # Más peso en clasificación para distinguir clases
        cls=1.5,
        box=7.5,
        
        # Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        # Optimizaciones
        cos_lr=True,
        close_mosaic=15,
        patience=50,
        warmup_epochs=3,
        
        cache="ram",
        workers=8,
        verbose=True,
        
        project=str(output_dir),
        name="speedreye_v3",
        
        # Guardar cada 10 épocas
        save_period=10,
    )

    best_weights = output_dir / "speedreye_v3" / "weights" / "best.pt"
    print(f"\n✅ Entrenamiento terminado. Mejor modelo: {best_weights}")

    models_dir.mkdir(parents=True, exist_ok=True)
    final_path = models_dir / output_name
    shutil.copy(best_weights, final_path)
    print(f"📦 Copiado a: {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Entrena detector V3 con datos en data/")
    parser.add_argument("--kitti-root", type=Path, default=KITTI_ROOT,
                        help=f"Ruta a KITTI (por defecto: {KITTI_ROOT})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output-name", type=str, default="speedreye_kitti_v3.pt")
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    # Verificar que existe KITTI
    if not args.kitti_root.exists():
        print(f"❌ No se encuentra KITTI en: {args.kitti_root}")
        print("   Asegúrate de que data/kitti_raw/training/ existe")
        return

    dataset_yaml = create_balanced_dataset(
        args.kitti_root, args.output_dir, args.val_ratio
    )

    train_with_checkpoints(
        dataset_yaml, args.model, args.imgsz, args.epochs,
        args.output_dir, args.models_dir, args.output_name
    )


if __name__ == "__main__":
    main()