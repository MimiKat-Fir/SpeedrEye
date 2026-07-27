# SpeedrEye — Resumen del pipeline (estado actual)

> Documento de contexto para el equipo. Objetivo: que cualquiera (persona o IA)
> entienda de un vistazo qué hace el proyecto, cómo está montado el código, y
> qué decisiones de diseño explican por qué está así.
>
> Basado en la exportación de `src/` del 2026-07-27. Si vuelves a exportar el
> proyecto y algo de aquí no coincide, confía en el código real, no en este
> documento — actualízalo cuando cambie algo importante.

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
Filtrado por horizonte (ver sección 6, calibración)
      │
      ▼
┌──────────────────┐  Geometría (altura_real × focal / altura_bbox_px)
│ distance/         │  + corrección aprendida (geometry_guided_v2.pt)
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
│ alert/system │   (trapecio frente a la cámara, ajustable a mano)? → alerta
└─────────────┘
      │
      ▼
┌──────────────┐  Bboxes, distancia, trayectoria, trapecio, banner de
│ visualizer.py │  alerta, panel de FPS/detecciones — todo escalado a la
└──────────────┘  resolución real del vídeo, ventana redimensionable.
```

Orquestado desde `main.py` (`SpeedrEyePipeline.process_frame`), que además
mide y guarda el tiempo de cada etapa en `results/performance/*.json`.

### Decisión de diseño clave: nada se ejecuta si no hay detecciones
Si un frame no tiene peatones/ciclistas, se saltan por completo: extracción
de pose, estimación de distancia, y evaluación por objeto de alertas. El
trapecio de alerta, en cambio, **se sigue dibujando siempre** (haya o no
detecciones) — es una referencia visual fija, no algo que deba desaparecer.

### Decisión de diseño clave: la recalibración de horizonte está desactivada
`calibration/geocalib.py` puede recalibrar el horizonte/punto de fuga cada
pocos frames usando Hough. Es costoso (intersección de líneas, aunque
acotado a las 40 más largas) y en la práctica puede producir valores
ruidosos si la escena tiene pocas líneas claras. `Config.
ENABLE_SCENE_RECALIBRATION = False` por defecto: se calibra una vez al
inicio del vídeo (cámara asumida fija) y no se vuelve a tocar.

El horizonte estimado automáticamente **no es 100% fiable por sí solo**
(depende de qué líneas detecte Hough en ese frame concreto, del pitch real
de la cámara, etc.), así que actualmente es la única barrera que filtra
detecciones espurias en la parte alta de la imagen — es un punto identificado
para mejorar (ver sección 7, "Próximos pasos").

## 3. Estructura de carpetas relevante

```
SpeedrEye/
├── src/pipeline/            # código de producción (ver arquitectura arriba)
├── notebooks/
│   ├── kitti_download.py    # descarga KITTI (imágenes+labels+calib) sin login
│   ├── train_detector_v2.py / v3  # entrena el detector YOLO
│   └── train_distance_head_v2.py  # entrena la cabecita de distancia
├── data/
│   ├── kitti_raw/            # KITTI descargado (creado automáticamente)
│   └── kitti_yolo_v3/        # KITTI convertido a formato YOLO, filtrado + sobremuestreado
├── models/
│   ├── yolo/                  # pesos del detector (.pt) — varias generaciones conviven
│   └── distance/               # cabecitas de corrección de distancia (.pt)
└── results/                    # calibración, métricas de rendimiento
```

## 4. Modelos actuales en producción

`src/pipeline/config.py` apunta a la generación más reciente de cada modelo:

```python
YOLO_MODEL_PATH = MODELS_DIR / "yolo" / "speedreye_kitti_v3.pt"
GEOMETRY_DISTANCE_WEIGHTS = MODELS_DIR / "distance" / "geometry_guided_v2.pt"
```

Generaciones anteriores (`speedreye_best.pt`, `speedreye_kitti_v2.pt`,
`geometry_guided_v1.pt`, `direct_distance.pt`) siguen en disco por si hace
falta comparar o volver atrás, pero no las usa el pipeline por defecto.

### Detector: `speedreye_kitti_v3.pt`

- Base: `yolo26n.pt` (preentrenado COCO), reentrenado **solo para detección**
  (clase + bbox) — sin LIDAR, sin orientación.
- Dataset: KITTI filtrado a `Pedestrian`/`Cyclist` (se descarta `Car`, `Van`,
  `Truck`, `Tram`, `Misc`, `DontCare`, `Person_sitting`), con **sobremuestreo
  de la clase minoritaria**: imágenes con ciclista ×3, imágenes con ambas
  clases ×2 — KITTI tiene muchos más peatones que ciclistas de origen.
- `imgsz=640`, `epochs=150` (completadas todas, `patience=50` no llegó a
  activarse), `batch=0.9` (AutoBatch de Ultralytics al 90% de VRAM libre, no
  un número fijo), `cls=1.5` / `box=7.5` (más peso a clasificación, para
  distinguir mejor Peatón/Ciclista), `cos_lr=True`, `mixup=0.1`,
  `cache="ram"`, `workers=8`.
- Split train/val por bloques de frames barajados (no aleatorio puro),
  `val_ratio≈0.15` → **594 imágenes / 1360 instancias de validación**.
- Checkpoint intermedio cada 10 épocas si mejora, además del `best.pt`
  habitual de Ultralytics.
- Tiempo total: **8.97 horas** (150 épocas).

**Resultado de validación final:**

| Clase | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Pedestrian | 0.899 | 0.779 | 0.888 | 0.564 |
| Cyclist | 0.935 | 0.905 | 0.967 | 0.729 |
| **Global** | **0.917** | **0.842** | **0.928** | **0.646** |

Ciclista rinde mejor que Peatón en todas las métricas — el sobremuestreo
compensó el desbalance de KITTI. El recall de Peatón (0.779) es el punto más
débil actual: es la clase que más se sigue "escapando" al modelo (más
variedad de tamaño/postura/oclusión que Ciclista).

### Cabecita de distancia: `geometry_guided_v2.pt`

No es un YOLO — es una MLP diminuta (`DistanceRegressionHead`, dos capas
`Linear`) que aprende un **factor de corrección multiplicativo** sobre una
estimación geométrica simple (`altura_real × focal / altura_bbox_px`), no la
distancia absoluta. El LIDAR de KITTI se usa solo offline, para generar la
etiqueta de entrenamiento (`distancia_real / distancia_geométrica`); nunca
se usa en producción.

Entrenada **congelando** el detector v3 (solo se capturan sus features vía
`forward_pre_hook`, sin actualizar ninguno de sus pesos). El checkpoint
guarda el hash SHA256 del detector con el que se entrenó — `load_distance_
head()` lo verifica al cargar, para no mezclar una cabecita con un detector
distinto al que la entrenó.

- `epochs=40` (mejor checkpoint en la época 28), `lr=1e-3` (Adam), batch=1
  (una imagen completa a la vez).
- Train/val: **6361 / 1120** imágenes.
- Mejor `val_loss`: **0.0049** (MSE sobre el logaritmo del factor de
  corrección — no es una distancia en metros directamente interpretable,
  la señal relevante es la tendencia, no el valor absoluto).
- Tiempo total: 2.31 horas.

**Importante:** cada vez que se reentrena el detector, hay que reentrenar
también esta cabecita — el check de hash SHA256 la invalida automáticamente
si detecta un detector distinto.

## 5. Scripts de entrenamiento (`notebooks/`)

- `kitti_download.py`: descarga `data_object_image_2.zip` (~12.6GB),
  `data_object_label_2.zip` y `data_object_calib.zip` directamente del
  bucket oficial de KITTI (sin login), los extrae en `data/kitti_raw/`. Se
  llama automáticamente desde los scripts de entrenamiento si falta.
- `train_detector_v2.py` / la variante v3 usada en el último entrenamiento:
  convierte KITTI a formato YOLO con el sobremuestreo descrito arriba,
  entrena, y copia el mejor checkpoint a `models/yolo/` con nombre
  versionado. Guarda checkpoints intermedios cada 10 épocas vía un callback
  de Ultralytics (`on_model_save`) si hay mejora de fitness.
- `train_distance_head_v2.py`: entrena la cabecita, reutilizando
  directamente `BackboneFeatureCapture`, `DistanceRegressionHead` y
  `prepare_distance_inputs` de `src/pipeline/distance/head.py` — así el
  checkpoint es garantizado compatible con el pipeline de producción.
- El script de detector puede encadenar automáticamente el de la cabecita
  al terminar (pensado para dejarlo corriendo toda la noche sin
  supervisión) — así se generó la pareja v3 + geometry_guided_v2 en una
  sola sesión nocturna (~11h20m total: 8h58m detector + 2h19m cabecita).

## 6. Otros módulos de producción

- **`calibration/geocalib.py`**: calibra focal/centro óptico con GeoCalib al
  inicio del vídeo (si el paquete no está instalado, avisa y usa una focal
  por defecto en vez de instalar nada en silencio). Detecta horizonte/punto
  de fuga vía Hough, acotado a las 40 líneas más largas y descartando pares
  casi paralelos (menos ruido, menos coste que comparar todas las líneas
  detectadas). Recalibración continua desactivada por defecto (sección 2).
  Ya no incluye `undistort_frame`/`camera_matrix` (no se usaban en ningún
  punto del pipeline) ni el antiguo `geocalib_loader.py` (código muerto,
  eliminado).
- **`tracking/pose_estimator.py`**: YOLOv8-pose ligero, cada
  `Config.POSE_FRAME_SKIP` frames, para estimar la orientación del torso vía
  keypoints. Se funde con la dirección de movimiento del Kalman (no la
  sustituye) para evitar parpadeos de dirección.
- **`tracking/kalman_predictor.py`**: un filtro de Kalman por `track_id`.
  Devuelve `None` (no genera trayectoria) si la velocidad estimada está por
  debajo de un umbral — evita proyectar flechas y evaluar alertas sobre
  objetos parados.
- **`alert/system.py`**: zona de peligro trapezoidal frente a la cámara
  (ajustable a mano al iniciar cada vídeo, con las esquinas inferiores
  arrancando en las esquinas reales del frame); alerta si algún punto de la
  trayectoria futura cae dentro.
- **`visualizer.py`**: todo el overlay (grosor de líneas, tamaño de texto,
  márgenes) escala con la resolución real del frame respecto a una
  referencia de 1280px — se ve igual de legible en 480p que en 4K. La
  alerta ya no se marca por caja individual: es el trapecio en rojo +
  un banner compacto arriba a la derecha, para no saturar la imagen con
  varias detecciones a la vez. Panel de estado (FPS/detecciones/tiempo de
  frame) fijo a la izquierda con contorno de texto para legibilidad sobre
  cualquier fondo.
- **`main.py`**: ventana redimensionable de verdad en Linux y Windows
  (`WINDOW_NORMAL | WINDOW_KEEPRATIO`, en vez del `WINDOW_AUTOSIZE` por
  defecto de `cv2.imshow`), dimensionada según la resolución real del
  vídeo al arrancar.

## 7. Próximos pasos identificados (discutidos, aún no aplicados en `src/`)

Estas ideas se plantearon a raíz de falsos positivos observados en la parte
alta de la imagen (zona donde el horizonte automático puede fallar) y de
querer robustecer el sistema de alerta. **Aún no están en el código actual**
— quedan aquí para no perderlas:

- **Color único para ambas clases** en vez de un color por clase: como
  distinguir Peatón/Ciclista a simple vista no aporta nada operativo (lo que
  importa es detectarlos), un único color evitaría la falsa sensación de
  "se confunden" cuando solo cambia el color de caja.
- **Zona de detección de clases ajustable** (un rectángulo, no trapecio, ya
  que aquí no hace falta modelar perspectiva): por defecto desde el
  horizonte hacia abajo, pero corregible a mano — soluciona de raíz que el
  horizonte automático sea la única barrera contra falsos positivos en la
  parte alta de la imagen.
- **Cruce de trayectoria más robusto en el trapecio de alerta**: comprobar
  no solo si algún punto muestreado de la predicción cae dentro del
  polígono, sino también si el segmento entre dos puntos consecutivos cruza
  alguno de sus lados — evita que una trayectoria "salte" la zona entre dos
  muestras sin que ninguna caiga literalmente dentro.
- Se descartó sustituir el trapecio por una línea horizontal a todo el
  ancho: perdería la confinación lateral (más ancho cerca de la cámara, más
  estrecho lejos) que evita alertas por objetos que nunca se acercan
  realmente a la trayectoria del vehículo.

## 8. Para reproducir el entrenamiento desde cero

```bash
# Paso 1: detector (descarga KITTI automáticamente si falta en data/kitti_raw)
python notebooks/train_detector_v2.py

# Paso 2: cabecita de distancia (usa el detector del paso 1)
# — se encadena automáticamente al terminar el paso 1, o se puede lanzar suelta:
python notebooks/train_distance_head_v2.py \
    --detector-weights models/yolo/speedreye_kitti_v3.pt
```