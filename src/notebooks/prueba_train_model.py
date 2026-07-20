#!/usr/bin/env python3
"""
SpeedrEye - Entrenamiento YOLO KITTI
Clases: Pedestrian, Cyclist
Versión para ejecución local (corregida)
"""

import os
import shutil
import random
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ============================================
# CONFIGURACIÓN
# ============================================

# Directorios (AJUSTA ESTAS RUTAS A TU SISTEMA)
BASE_DIR = Path(__file__).parent  # Directorio donde está este script
DATA_DIR = BASE_DIR / "data" / "kitti_raw"
OUT_DIR = BASE_DIR / "data" / "kitti_filtered"
WEIGHTS_DIR = BASE_DIR / "weights26"
LOGS_DIR = BASE_DIR / "logs26"
CONFIGS_DIR = BASE_DIR / "configs"

# Crear directorios
for d in [DATA_DIR, OUT_DIR, WEIGHTS_DIR, LOGS_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Clases que nos interesan (SOLO peatones y bicis)
TARGET_CLASSES = {
    "Pedestrian": 0,
    "Cyclist": 1,
}

CLASS_NAMES = {v: k for k, v in TARGET_CLASSES.items()}
NUM_CLASSES = len(TARGET_CLASSES)

# Parámetros de entrenamiento (validados para Ultralytics)
MODEL_NAME = "yolo26n.pt"  # o "yolo26n.pt" si lo tienes
EPOCHS = 100
IMAGE_SIZE = 832
BATCH_SIZE = 8  # Para GPU 4GB
DEVICE = 0  # 0 = GPU, "cpu" = CPU
WORKERS = 2

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     SPEEDREYE - ENTRENAMIENTO                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Clases: {list(TARGET_CLASSES.keys())}                         ║
║  Modelo: {MODEL_NAME}                                          ║
║  Épocas: {EPOCHS}                                              ║
║  Batch size: {BATCH_SIZE}                                      ║
║  Device: {'GPU' if DEVICE == 0 else 'CPU'}                    ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ============================================
# 1. DESCARGAR KITTI (solo si no existe)
# ============================================

def download_kitti():
    """Descarga KITTI si no está presente"""
    zip_images = DATA_DIR / "data_object_image_2.zip"
    zip_labels = DATA_DIR / "data_object_label_2.zip"
    
    # Verificar si ya está descargado y descomprimido
    training_dir = DATA_DIR / "training"
    if training_dir.exists() and (training_dir / "image_2").exists():
        print("✅ KITTI ya está descargado y descomprimido")
        return
    
    if zip_images.exists() and zip_labels.exists():
        print("✅ Archivos ZIP ya descargados, descomprimiendo...")
        with zipfile.ZipFile(zip_images, 'r') as z:
            z.extractall(DATA_DIR)
        with zipfile.ZipFile(zip_labels, 'r') as z:
            z.extractall(DATA_DIR)
        print("✅ Descompresión completada")
        return
    
    print("📂 Descargando KITTI...")
    print("   (Esto puede tomar varios minutos)")
    
    import subprocess
    subprocess.run([
        "wget", "-q", "--show-progress",
        "-P", str(DATA_DIR),
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    ], check=True)
    
    subprocess.run([
        "wget", "-q", "--show-progress",
        "-P", str(DATA_DIR),
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    ], check=True)
    
    # Descomprimir
    print("📦 Descomprimiendo...")
    with zipfile.ZipFile(zip_images, 'r') as z:
        z.extractall(DATA_DIR)
    with zipfile.ZipFile(zip_labels, 'r') as z:
        z.extractall(DATA_DIR)
    
    print("✅ KITTI descargado y descomprimido")

# ============================================
# 2. FILTRAR CLASES
# ============================================

def filter_dataset():
    """Filtra solo imágenes con peatones y ciclistas"""
    raw_image_dir = DATA_DIR / "training" / "image_2"
    raw_label_dir = DATA_DIR / "training" / "label_2"
    
    if not raw_image_dir.exists():
        print("❌ No se encontraron datos KITTI. Ejecuta download_kitti() primero")
        return []
    
    print("📋 Filtrando dataset...")
    
    filtered = []
    image_files = list(raw_image_dir.glob("*.png"))
    
    for img_path in tqdm(image_files, desc="Procesando"):
        label_path = raw_label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        valid_objects = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_name = parts[0]
            if cls_name in TARGET_CLASSES:
                valid_objects.append(parts)
        
        if valid_objects:
            filtered.append({
                'image': img_path,
                'objects': valid_objects
            })
    
    print(f"✅ Encontradas {len(filtered)} imágenes con viandantes")
    return filtered

# ============================================
# 3. DIVIDIR Y CONVERTIR
# ============================================

def split_and_convert(filtered_data):
    """Divide en train/val/test y convierte a formato YOLO"""
    
    # Crear estructura
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            (OUT_DIR / sub / split).mkdir(parents=True, exist_ok=True)
    
    # Dividir
    random.seed(42)
    random.shuffle(filtered_data)
    
    total = len(filtered_data)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    
    splits = {
        "train": filtered_data[:train_end],
        "val": filtered_data[train_end:val_end],
        "test": filtered_data[val_end:]
    }
    
    print("📊 División:")
    for name, data in splits.items():
        print(f"   {name}: {len(data)} imágenes")
    
    # Convertir
    for split_name, data in splits.items():
        print(f"🔄 Convirtiendo {split_name}...")
        
        for item in tqdm(data, desc=f"  {split_name}"):
            img_path = item['image']
            objects = item['objects']
            
            # Tamaño de imagen
            with Image.open(img_path) as img:
                w, h = img.size
            
            # Copiar imagen
            dst_img = OUT_DIR / "images" / split_name / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Crear etiquetas YOLO
            yolo_lines = []
            for obj in objects:
                cls_name = obj[0]
                cls_id = TARGET_CLASSES[cls_name]
                
                xmin = float(obj[4])
                ymin = float(obj[5])
                xmax = float(obj[6])
                ymax = float(obj[7])
                
                xc = ((xmin + xmax) / 2) / w
                yc = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            
            dst_label = OUT_DIR / "labels" / split_name / f"{img_path.stem}.txt"
            dst_label.write_text("\n".join(yolo_lines))
    
    return splits

# ============================================
# 4. CREAR CONFIGURACIÓN YOLO
# ============================================

def create_yolo_config():
    """Crea el archivo de configuración YOLO"""
    yaml_content = f"""
# SpeedrEye - KITTI Viandantes
# Clases: {list(TARGET_CLASSES.keys())}

path: {OUT_DIR}
train: images/train
val: images/val
test: images/test

names:
"""
    for cls_id, cls_name in CLASS_NAMES.items():
        yaml_content += f"  {cls_id}: {cls_name}\n"
    yaml_content += f"\nnc: {NUM_CLASSES}\n"
    
    config_path = CONFIGS_DIR / "kitti.yaml"
    config_path.write_text(yaml_content)
    
    print(f"📋 Configuración guardada: {config_path}")
    return config_path

# ============================================
# 5. ENTRENAR (CORREGIDO)
# ============================================

def train_model(config_path):
    """Entrena el modelo YOLO"""
    print("\n🚀 Iniciando entrenamiento...")
    
    model = YOLO(MODEL_NAME)
    
    # Parámetros válidos para Ultralytics (eliminados los no válidos)
    results = model.train(
        data=str(config_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(LOGS_DIR),
        name="speedreye_train",
        exist_ok=True,
        save=True,
        save_period=10,  # Guardar cada 10 épocas (válido)
        plots=True,
        # val_period eliminado (no es válido en esta versión)
        verbose=True
    )
    
    # Guardar mejores pesos
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        shutil.copy2(best_path, WEIGHTS_DIR / "speedreye_best.pt")
        print(f"🏆 Modelo guardado: {WEIGHTS_DIR}/speedreye_best.pt")
    
    # Guardar último modelo
    last_path = Path(results.save_dir) / "weights" / "last.pt"
    if last_path.exists():
        shutil.copy2(last_path, WEIGHTS_DIR / "speedreye_last.pt")
        print(f"📁 Último modelo guardado: {WEIGHTS_DIR}/speedreye_last.pt")
    
    return results

# ============================================
# 6. EVALUAR
# ============================================

def evaluate_model(config_path):
    """Evalúa el modelo entrenado"""
    model_path = WEIGHTS_DIR / "speedreye_best.pt"
    
    if not model_path.exists():
        print("❌ No se encontró modelo para evaluar")
        return
    
    print("\n📊 Evaluando modelo...")
    model = YOLO(str(model_path))
    
    metrics = model.val(
        data=str(config_path),
        split='val',
        batch=8,
        workers=2,
        plots=True
    )
    
    print("\n📊 Resultados:")
    print("="*40)
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("="*40)

# ============================================
# 7. MAIN
# ============================================

def main():
    """Ejecuta todo el pipeline"""
    
    # 1. Descargar KITTI
    download_kitti()
    
    # 2. Filtrar dataset
    filtered = filter_dataset()
    if not filtered:
        print("❌ No se encontraron datos. Verifica la descarga.")
        return
    
    # 3. Dividir y convertir
    splits = split_and_convert(filtered)
    
    # 4. Crear configuración
    config_path = create_yolo_config()
    
    # 5. Entrenar
    train_model(config_path)
    
    # 6. Evaluar
    evaluate_model(config_path)
    
    print("\n✅ Pipeline completado")
    print(f"   Modelo en: {WEIGHTS_DIR}/speedreye_best.pt")
    print(f"   Dataset en: {OUT_DIR}")

if __name__ == "__main__":
    main()