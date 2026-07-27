#!/usr/bin/env python3
"""
Orquestador para entrenamiento nocturno completo:
1. Entrena detector V3 (150 épocas) usando datos en data/
2. Automáticamente entrena cabeza de distancia V2
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {description} completado en {elapsed/60:.1f} minutos")
    else:
        print(f"\n❌ {description} falló con código {result.returncode}")
        sys.exit(1)
    
    return result

def main():
    print("\n" + "="*60)
    print("🌙 ENTRENAMIENTO NOCTURNO SPEEDREYE")
    print("="*60)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"KITTI: {PROJECT_ROOT / 'data' / 'kitti_raw'}")
    print("="*60 + "\n")
    
    # Verificar que KITTI existe
    kitti_path = PROJECT_ROOT / "data" / "kitti_raw" / "training"
    if not kitti_path.exists():
        print(f"❌ No se encuentra KITTI en: {kitti_path}")
        print("   Asegúrate de tener data/kitti_raw/training/")
        sys.exit(1)
    
    # Paso 1: Entrenar detector V3
    detector_cmd = (
        f"python {PROJECT_ROOT}/train_detector_v3.py "
        f"--kitti-root {PROJECT_ROOT}/data/kitti_raw "
        f"--output-dir {PROJECT_ROOT}/data/kitti_yolo_v3 "
        f"--output-name speedreye_kitti_v3.pt "
        f"--epochs 150 "
        f"--imgsz 640 "
        f"--model yolo26n.pt"
    )
    
    run_command(detector_cmd, "Entrenando Detector V3 (150 épocas)")
    
    # Verificar que el detector se generó
    detector_path = PROJECT_ROOT / "models" / "yolo" / "speedreye_kitti_v3.pt"
    if not detector_path.exists():
        print(f"❌ No se encontró el detector en: {detector_path}")
        sys.exit(1)
    
    # Paso 2: Entrenar cabeza de distancia para V3
    distance_cmd = (
        f"python {PROJECT_ROOT}/train_distance_head_v2.py "
        f"--kitti-root {PROJECT_ROOT}/data/kitti_raw "
        f"--detector-weights {detector_path} "
        f"--output {PROJECT_ROOT}/models/distance/geometry_guided_v2.pt "
        f"--epochs 40"
    )
    
    run_command(distance_cmd, "Entrenando Cabeza de Distancia V2 (40 épocas)")
    
    print("\n" + "="*60)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*60)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📦 Modelos generados:")
    print(f"   Detector:      models/yolo/speedreye_kitti_v3.pt")
    print(f"   Cabeza dist.:  models/distance/geometry_guided_v2.pt")

if __name__ == "__main__":
    main()