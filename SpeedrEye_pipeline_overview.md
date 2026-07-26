# SpeedrEye — Resumen del pipeline (estado actual)

> Documento de contexto para el equipo. Objetivo: que cualquiera (persona o IA)
> entienda de un vistazo qué hace el proyecto, cómo está montado el código, y
> qué decisiones de diseño explican por qué está así.

## 1. Qué es SpeedrEye

Sistema de detección y alerta de colisión en tiempo real para peatones y
ciclistas desde una cámara frontal (tipo dashcam / vehículo). Por cada frame
de vídeo:

1. Detecta peatones y ciclistas.
2. Estima a qué distancia están (en metros).
3. Predice hacia dónde se van a mover en el próximo segundo.
4. Si esa trayectoria futura entra en una "zona de peligro" frente a la
   cámara, dispara una alerta visual.

Todo tiene que correr en tiempo real (objetivo: muy por debajo del tiempo de
reacción humano, ~200-250ms/frame), en hardware modesto (se ha probado en una
GTX 1650 de 4GB).

## 2. Arquitectura del pipeline (`src/pipeline/`)

```
Frame de vídeo
      │
      ▼
┌─────────────┐   YOLO (detección + tracking ByteTrack)
│ detector.py │   → clase, bbox, track_id, confianza
└─────────────┘
      │
      ▼
┌──────────────────┐  Geometría (altura_real × focal / altura_bbox_px)
│ distance/         │  + corrección aprendida opcional (ver sección 4)
│  geometry_guided  │  → distancia en metros por objeto
└──────────────────┘
      │
      ▼
┌───────────────────┐  Kalman filtra posición (x,z) por track_id.
│ tracking/          │  Si hay pose disponible (cada N frames), funde
│  kalman_predictor  │  dirección de movimiento + orientación del torso.
│  pose_estimator     │  Si el objeto está quieto, NO genera trayectoria
└───────────────────┘  (ahorra cómputo y evita falsas alertas).
      │
      ▼
┌─────────────┐   ¿La trayectoria futura entra en la zona de peligro
│ alert/system │   (trapecio frente a la cámara)? → alerta sí/no
└─────────────┘
      │
      ▼
┌──────────────┐
│ visualizer.py │  Dibuja bboxes, distancia, trayectoria, zona, alerta, FPS
└──────────────┘
```

Orquestado desde `main.py` (`SpeedrEyePipeline.process_frame`), que además
mide y guarda el tiempo de cada etapa en `results/performance/*.json`.

### Decisión de diseño clave: nada se ejecuta si no hay detecciones
Si un frame no tiene peatones/ciclistas, se saltan por completo: extracción
de pose, estimación de distancia, y evaluación de alertas. Antes se ejecutaba
igual "en vacío", desperdiciando tiempo de frame sin ningún beneficio.

### Decisión de diseño clave: la recalibración de horizonte está desactivada
`calibration/geocalib.py` puede recalibrar el horizonte/punto de fuga cada
pocos frames usando Hough. Se detectó que esto era muy costoso (intersección
de todas las combinaciones de líneas detectadas, O(n²)) y que en la práctica
producía valores muy ruidosos (saltos de horizonte de cientos de píxeles
entre frames), causando micro-congelaciones periódicas del vídeo y filtrando
detecciones válidas por error. Ahora `Config.ENABLE_SCENE_RECALIBRATION =
False` por defecto: se calibra una vez al inicio del vídeo (cámara asumida
fija) y no se vuelve a tocar.

## 3. Estructura de carpetas relevante

```
SpeedrEye/
├── src/pipeline/           # código de producción (ver arquitectura arriba)
├── notebooks/
│   ├── kitti_download.py   # descarga KITTI (imágenes+labels+calib) sin login
│   ├── train_detector.py   # entrena el detector YOLO (Paso 1, ver sección 4)
│   └── train_distance_head.py  # entrena la cabecita de distancia (Paso 2)
├── data/
│   ├── kitti_raw/           # KITTI descargado (creado automáticamente)
│   └── kitti_yolo/          # KITTI convertido a formato YOLO, filtrado
├── models/
│   ├── yolo/                 # pesos del detector (.pt)
│   └── distance/             # cabecitas de corrección de distancia (.pt)
└── results/                  # calibración, métricas de rendimiento
```

## 4. Cómo se entrenan los modelos (y por qué así)

**Contexto importante:** hubo un intento previo de entrenar un único YOLO
para que devolviera a la vez clase + distancia + orientación 3D, usando
KITTI completo (imágenes + LIDAR). Resultado: la detección empeoró mucho
(perdía objetos delante de la cámara) y las distancias no eran fiables. Causa
más probable: **interferencia negativa entre tareas** al compartir backbone
— la loss de distancia/orientación "tira" de las features en una dirección
que perjudica a la detección pura. Se volvió al enfoque de **dos modelos
separados**, que es el que hay ahora mismo:

### Paso 1 — Detector (`train_detector.py`)
- YOLO26n, entrenado **solo para detección** (clase + bbox), sobre KITTI
  filtrado a `Pedestrian` y `Cyclist` únicamente (se descarta `Car`, `Van`,
  `Truck`, `Tram`, `Misc`, `DontCare`, y también `Person_sitting` — la
  geometría del paso 2 asume persona de pie).
- Sin LIDAR, sin orientación. Es exactamente el tipo de entrenamiento que ya
  había dado buenos resultados antes.
- Split train/val por bloques contiguos de frames (no aleatorio puro), para
  reducir fuga de información entre frames casi idénticos del mismo trayecto.
- `batch=-1` (AutoBatch de Ultralytics, se adapta a la VRAM disponible —
  necesario en GPUs pequeñas).

### Paso 2 — Corrección de distancia (`train_distance_head.py`)
**No es otro YOLO.** Es una cabecita de regresión minúscula (2 capas
`Linear`, ver `src/pipeline/distance/head.py::DistanceRegressionHead`) que:

1. Parte de una estimación geométrica simple, sin red neuronal:
   `distancia_geom = altura_real_de_la_clase × focal / altura_bbox_en_píxeles`
2. Aprende solo un **factor de corrección multiplicativo** sobre esa
   estimación: `distancia_final = distancia_geom × factor_aprendido`.
3. El factor objetivo de entrenamiento sale de
   `distancia_real_KITTI / distancia_geom`, donde la "distancia real" es el
   campo `location.z` de las etiquetas 3D de KITTI (construido usando LIDAR
   **por KITTI, offline**, al crear el dataset).

**El LIDAR nunca se usa en producción**, ni por el detector ni por la
cabecita — solo sirve, una vez, para generar la etiqueta de entrenamiento
de la cabecita. En inferencia, el pipeline solo necesita la imagen RGB y la
focal de la cámara (calibrada con GeoCalib).

Por qué corrección sobre geometría y no regresión directa: predecir un
factor cercano a 1.0 es una tarea mucho más fácil y estable de aprender que
predecir la distancia absoluta desde cero (lo que hacía el enfoque anterior,
`distance/direct.py`, y explicaba en parte por qué las distancias no eran ni
razonables). Además, al depender de la focal real calibrada en vez de una
escala aprendida implícitamente del dominio de cámara de KITTI, generaliza
mejor a cámaras distintas a las de KITTI.

El detector se usa **congelado** al entrenar esta cabecita (no se actualiza
ningún peso suyo) — solo se capturan sus features intermedias vía un
`forward_pre_hook` (`BackboneFeatureCapture`) para pasarlas por ROI-align.
El checkpoint resultante guarda el hash SHA256 del detector con el que se
entrenó, y `load_distance_head()` lo verifica al cargar en producción, para
evitar mezclar una cabecita con un detector distinto al que la entrenó.

## 5. Otros módulos de producción

- **`tracking/pose_estimator.py`**: YOLOv8-pose ligero, cada
  `Config.POSE_FRAME_SKIP` frames, para estimar la orientación del torso vía
  keypoints. Se funde con la dirección de movimiento del Kalman (no la
  sustituye) para evitar parpadeos de dirección.
- **`tracking/kalman_predictor.py`**: un filtro de Kalman por `track_id`.
  Devuelve `None` (no genera trayectoria) si la velocidad estimada está por
  debajo de un umbral — evita proyectar flechas y evaluar alertas sobre
  objetos que están parados.
- **`alert/system.py`**: zona de peligro trapezoidal frente a la cámara;
  alerta si algún punto de la trayectoria futura cae dentro.
- **`calibration/geocalib.py`**: calibra focal/centro óptico con GeoCalib al
  inicio del vídeo; detecta horizonte/punto de fuga vía Hough (recalibración
  continua desactivada, ver sección 2).

## 6. Estado actual (a fecha de este documento)

- Detector `speedreye_kitti_v2.pt` en proceso/recién entrenado sobre KITTI
  (solo detección, YOLO26n, ~80-150 épocas). Resultados de validación en
  torno a mAP50≈0.85, mAP50-95≈0.53 en el punto donde se revisó.
- Pendiente: entrenar `geometry_guided_v1.pt` (la cabecita) usando ese
  detector ya entrenado.
- Pendiente: actualizar `Config.YOLO_MODEL_PATH` en `src/pipeline/config.py`
  para apuntar al nuevo detector una vez validado sobre vídeo real (no solo
  métricas de KITTI, que no garantizan el mismo comportamiento fuera de su
  dominio de cámara).

## 7. Para reproducir el entrenamiento desde cero

```bash
# Paso 1: detector (descarga KITTI automáticamente si falta en data/kitti_raw)
python notebooks/train_detector.py

# Paso 2: cabecita de distancia (usa el detector del paso 1)
python notebooks/train_distance_head.py \
    --detector-weights models/yolo/speedreye_kitti_v2.pt
```
