# Resultados para la presentacion

## Comparacion principal

| Metrica | YOLO26n original | YOLO26n ajustado a KITTI (v3) | SpeedrEye completo |
|---|---:|---:|---:|
| Clases de salida | 80 clases COCO | 2: peaton y ciclista | 2 + distancia por objeto |
| Parametros del detector | 2,572,280 | 2,504,580 | 2,509,061 |
| Tamano de pesos | 5.29 MB | 5.12 MB | 5.14 MB |
| Precision en KITTI | No medida en el mismo split | **0.917** | **0.917** |
| Recall en KITTI | No medido en el mismo split | **0.841** | **0.841** |
| mAP50 en KITTI | No medido en el mismo split | **0.928** | **0.928** |
| mAP50-95 en KITTI | No medido en el mismo split | **0.646** | **0.646** |
| Distancia en metros | No | No | Si |
| Latencia CPU del detector* | 1,141.9 ms | 1,203.0 ms | 1,208.2 ms aprox. |
| FPS CPU medidos* | 0.876 | 0.831 | 0.828 aprox. |

El detector ajustado tiene un 2.63% menos de parametros que el original porque
su salida se especializa en dos clases. La cabeza de distancia solo anade 4,481
parametros y 20.9 KB.

## Mejora entre versiones ajustadas

| Modelo | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Primer ajuste (`speedreye_best.pt`) | 0.918 | 0.791 | 0.878 | 0.556 |
| KITTI v2 | 0.865 | 0.757 | 0.863 | 0.542 |
| **KITTI v3 final** | **0.917** | **0.841** | **0.928** | **0.646** |
| Mejora v2 -> v3 | +6.0% | +11.2% | +7.5% | **+19.2%** |

La mejora mas importante de v3 es el mAP50-95. Esto indica cajas mas precisas
en un rango exigente de umbrales IoU, no solo mas detecciones faciles.

## Resultado por clase del modelo final

| Clase KITTI | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Peaton | 0.899 | 0.779 | 0.888 | 0.564 |
| Ciclista | **0.935** | **0.905** | **0.967** | **0.729** |
| **Global** | **0.917** | **0.842** | **0.928** | **0.646** |

El ciclista es la clase mas fuerte. El siguiente margen de mejora esta en el
recall de peatones: aproximadamente un 22% no se detecta en el conjunto de
validacion al umbral usado por la evaluacion.

## Distancia y coste computacional

| Dato | Resultado |
|---|---:|
| Parametros extra de la cabeza de distancia | 4,481 |
| Incremento de parametros frente al detector | **0.18%** |
| Tamano extra de pesos | 20.9 KB |
| Latencia media de distancia* | **5.17 ms/frame** |
| Sobrecoste frente a deteccion + tracking* | **0.47%** |
| Error del primer modelo de distancia directa | MAE **1.00 m**, RMSE **1.56 m** |
| Modelo de distancia actual | Geometria + correccion aprendida |

El error en metros corresponde al primer modelo directo y no al modelo
geometrico actual. El modelo actual necesita una evaluacion separada contra las
distancias reales de KITTI antes de presentar un MAE propio.

## Coste del entrenamiento final

| Etapa | Datos | Epocas | Tiempo |
|---|---:|---:|---:|
| Detector KITTI v3 | 594 imagenes / 1,360 objetos en validacion | 150 | 8.97 h |
| Cabeza de distancia v2 | 6,361 train / 1,120 validacion | 40 (mejor: 28) | 2.31 h |
| **Total** | Entrenamiento nocturno en GPU | 190 | **11.28 h** |

## Mensajes utiles para las diapositivas

- El ajuste a KITTI alcanzo **92.8% mAP50** para peatones y ciclistas.
- La version v3 mejoro el **mAP50-95 un 19.2%** respecto a v2.
- Se obtiene una distancia por deteccion con solo **0.18% mas parametros**.
- La cabeza de distancia supuso menos de **0.5% de sobrecoste** en la prueba CPU.
- El modelo completo ocupa aproximadamente **5.14 MB**, adecuado para exportar
  y probar posteriormente en un telefono.

## Metodologia y limites

Los resultados KITTI proceden de las metricas guardadas en los checkpoints y
del resumen de entrenamiento del repositorio. Las versiones no usaron
necesariamente un split identico, por lo que la evolucion entre checkpoints es
orientativa.

La prueba de velocidad uso 20 frames consecutivos (40-59) de
`videos/peatones-bici.mp4`, entrada YOLO de 640 px, confianza 0.4, dos
calentamientos y CPU Snapdragon X Elite sin CUDA. El video fuente es 854x480.
Los FPS son una referencia local, no una estimacion del rendimiento movil. En
el telefono final deben repetirse con el modelo exportado y el runtime elegido.

No se debe comparar el mAP COCO almacenado en el YOLO original con el mAP KITTI
del modelo ajustado: son datasets y tareas diferentes. Para una comparacion
academicamente valida falta ejecutar ambos modelos sobre el mismo split KITTI,
con la misma conversion de clases y los mismos umbrales.
