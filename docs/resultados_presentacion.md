# Resultados para la presentacion

## Resultado principal: YOLO original frente a SpeedrEye

Prueba local sobre 100 imagenes KITTI que no aparecen en el conjunto de
entrenamiento: 248 peatones y 17 ciclistas.

| Metrica | YOLO26n original | SpeedrEye KITTI v3 | Mejora |
|---|---:|---:|---:|
| Precision | 37.9% | **92.6%** | **+54.7 puntos** |
| Recall | 25.8% | **76.7%** | **+50.9 puntos** |
| mAP50 | 28.8% | **87.8%** | **+59.0 puntos / 3.05x** |
| mAP50-95 | 12.8% | **60.2%** | **+47.4 puntos / 4.71x** |
| Inferencia CPU | 620.3 ms/imagen | **597.9 ms/imagen** | **3.6% mas rapido** |

La conclusion defendible es que especializar YOLO en peatones y ciclistas de
carretera triplica el mAP50 sin aumentar el coste de inferencia.

## Resultado por clase

| Clase | mAP50 YOLO original | mAP50 SpeedrEye | Mejora absoluta |
|---|---:|---:|---:|
| Peaton | 57.4% | **88.2%** | **+30.8 puntos** |
| Ciclista | 0.1% | **87.3%** | **+87.2 puntos** |

El YOLO original reconoce personas razonablemente, pero no representa bien un
ciclista KITTI como una sola caja: suele separar persona y bicicleta. El ajuste
aprende directamente la clase y la caja que necesita la aplicacion.

## Que significa mAP50

Una deteccion se considera correcta cuando:

1. Predice la clase correcta.
2. Su caja coincide al menos un 50% con la caja real, medido mediante IoU.

La precision media se calcula para cada clase mientras cambia el umbral de
confianza. `mAP50` es la media de esas precisiones usando IoU 0.50.

- `100%`: cajas y clases casi siempre correctas.
- `0%`: el modelo no localiza correctamente los objetos.
- `mAP50-95` es mas exigente: repite la evaluacion desde IoU 0.50 hasta 0.95.
  Por eso mide mejor la calidad exacta de las cajas.

Ejemplo para explicar oralmente: si la caja predicha y la caja real se solapan
suficientemente, cuenta como acierto; mAP resume aciertos, falsos positivos y
objetos no encontrados en una sola metrica.

## Comparacion de metodos de distancia

| Metodo | Salida | Parametros extra | Tiempo extra CPU* | Error disponible | Decision |
|---|---|---:|---:|---:|---|
| Geometria de camara | Distancia por objeto | 0 | **0.06 ms** | Sin MAE validado | Baseline minimo |
| Regresion directa | Distancia aprendida por objeto | 4,481 | **6.62 ms** | **MAE 1.00 m; RMSE 1.56 m** | Comparacion entrenada |
| Geometria + correccion aprendida | Distancia por objeto | 4,481 | **5.17 ms** | Pendiente de MAE en metros | Metodo actual |
| MiDaS + pseudo-LiDAR | Mapa denso + nube de puntos | 21.32 M | **578.4 ms** | Profundidad relativa, no metrica por defecto | Descartado |

`*` Snapdragon X Elite, CPU sin CUDA. Los tiempos de las cabezas se midieron
sobre 20 frames de `peatones-bici.mp4`. La nube pseudo-LiDAR se midio con ocho
cajas y 1,615 puntos.

### Coste evitado al eliminar MiDaS

| Comparacion | Resultado |
|---|---:|
| Tiempo MiDaS + nube frente a cabeza actual | **112 veces mayor** |
| Parametros MiDaS frente a cabeza actual | **4,758 veces mas** |
| Memoria de parametros MiDaS FP32 | **85.3 MB** |
| Tamano de la cabeza actual | **20.9 KB** |
| Perdida estimada de FPS al anadir MiDaS al detector* | **34.3%** |

MiDaS procesaba todos los pixeles para crear un mapa de profundidad y despues
el pseudo-LiDAR convertia regiones del mapa en puntos 3D. SpeedrEye solo calcula
el numero que necesita para cada objeto detectado. Esta es la mejora de
arquitectura mas importante para una futura ejecucion movil.

## Coste del modelo y entrenamiento

| Dato | Resultado |
|---|---:|
| Parametros YOLO26n original | 2.57 M |
| Parametros detector SpeedrEye | 2.50 M |
| Peso detector SpeedrEye | 5.12 MB |
| Incremento por cabeza de distancia | **0.18%** |
| Entrenamiento detector v3 | 150 epocas / 8.97 h |
| Entrenamiento distancia v2 | 40 epocas / 2.31 h |

## Texto corto para una diapositiva

> Al adaptar YOLO al dominio KITTI, el mAP50 aumento del 28.8% al 87.8%:
> tres veces mejor, con un 3.6% menos de tiempo de inferencia en esta prueba.
> Ademas, sustituimos MiDaS y la nube pseudo-LiDAR por una cabeza de distancia
> 4,758 veces mas pequena y 112 veces mas rapida.

## Metodologia y limites

- Modelos: `yolo26n.pt` y `speedreye_kitti_v3.pt`.
- Evaluacion Ultralytics identica: 640 px, batch 4, CPU, clases 0 y 1.
- En el modelo original se interpreta `person` como Peaton y `bicycle` como
  Ciclista. Esta diferencia semantica forma parte del problema de dominio.
- Muestra limpia: 100 imagenes reconstruidas mediante la semilla 42 del script
  v3 y excluidas de sus identificadores de entrenamiento.
- Solo hay 17 ciclistas en esta prueba. El resultado por clase es util, pero
  debe confirmarse con un conjunto limpio mayor.
- El script v3 original sobremuestrea antes de dividir. Esto introduce 367
  imagenes repetidas entre sus 594 imagenes unicas de validacion y entrenamiento.
  Por esa razon, los valores guardados de 92.8% mAP50 y 64.6% mAP50-95 no se
  usan como comparacion principal.
- El MAE de 1.00 m pertenece al modelo de regresion directa anterior. La
  geometria corregida v2 necesita una evaluacion propia en metros.
- Los tiempos sirven para comparar metodos en el mismo PC. El FPS final debe
  medirse en el telefono y con el formato de despliegue definitivo.
