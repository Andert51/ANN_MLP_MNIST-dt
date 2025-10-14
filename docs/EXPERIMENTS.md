# 📊 Experimentos y Resultados

## Descripción de Experimentos

Este documento describe los experimentos realizados y los resultados obtenidos.

## Estructura de Experimentos

### 1. Configuración de Capas Ocultas

**Objetivo**: Determinar el impacto del número y tamaño de capas ocultas.

**Configuraciones probadas**:
- 1 capa: [32], [64], [128]
- 2 capas: [64, 32], [128, 64], [256, 128]
- 3 capas: [128, 64, 32], [256, 128, 64]
- 4+ capas: [512, 256, 128, 64]

**Métricas evaluadas**:
- Precisión de entrenamiento
- Precisión de prueba
- Tiempo de entrenamiento
- Número de épocas hasta convergencia

**Hipótesis**:
- Más capas = mayor capacidad de aprendizaje
- Riesgo de overfitting con demasiadas capas
- Trade-off entre precisión y tiempo de entrenamiento

### 2. Tasa de Aprendizaje

**Objetivo**: Encontrar la tasa de aprendizaje óptima.

**Valores probados**:
- 0.001 (muy bajo)
- 0.01 (bajo)
- 0.1 (medio)
- 0.5 (alto)
- 0.75 (muy alto)
- 1.0 (extremo)

**Métricas evaluadas**:
- Velocidad de convergencia
- Estabilidad del entrenamiento
- Precisión final

**Hipótesis**:
- LR muy bajo = convergencia lenta
- LR muy alto = inestabilidad, no converge
- LR óptimo = balance entre velocidad y estabilidad

### 3. Funciones de Activación

**Objetivo**: Comparar diferentes funciones de activación.

**Funciones probadas**:
- Sigmoid: clásica, range [0, 1]
- Tanh: range [-1, 1], mejor gradiente
- ReLU: más rápida, evita vanishing gradient

**Métricas evaluadas**:
- Velocidad de convergencia
- Precisión final
- Estabilidad numérica

### 4. Robustez al Ruido

**Objetivo**: Evaluar la resistencia del modelo a datos corruptos.

**Tipos de ruido**:
- Gaussiano: ruido aleatorio normal
- Salt & Pepper: píxeles blancos/negros aleatorios
- Speckle: ruido multiplicativo
- Uniforme: ruido distribuido uniformemente

**Niveles probados**: 0.05, 0.1, 0.15, 0.2

**Métricas evaluadas**:
- Precisión en datos limpios
- Precisión en datos ruidosos
- Degradación de rendimiento
- Puntuación de robustez (noisy_acc / clean_acc)

## Resultados Esperados

### Configuración Óptima Esperada

Basado en la literatura y experimentos previos:

```
Arquitectura óptima:
- Capas ocultas: [128, 64] o [256, 128]
- Tasa de aprendizaje: 0.01 - 0.1
- Función de activación: Tanh o ReLU
- Batch size: 64
- Épocas: 50-100

Rendimiento esperado:
- Precisión de entrenamiento: 98-99%
- Precisión de prueba: 95-97%
- Tiempo de entrenamiento: 30-120 segundos
```

### Métricas de Éxito

- **Excelente**: Test accuracy > 95%
- **Bueno**: Test accuracy 90-95%
- **Aceptable**: Test accuracy 85-90%
- **Necesita mejora**: Test accuracy < 85%

### Generalization Gap

- **Excelente**: Gap < 2%
- **Bueno**: Gap 2-5%
- **Overfitting**: Gap > 5%

## Análisis de Trade-offs

### 1. Complejidad vs Rendimiento

```
Modelo Simple [64]:
+ Rápido de entrenar
+ Menos parámetros
- Menor capacidad

Modelo Profundo [512, 256, 128, 64]:
+ Mayor capacidad
+ Mejor precisión potencial
- Más lento
- Riesgo de overfitting
```

### 2. Velocidad vs Precisión

```
Alta LR (0.5-1.0):
+ Convergencia rápida
- Puede oscilar
- Puede no encontrar óptimo

Baja LR (0.001-0.01):
+ Convergencia estable
+ Mejor precisión final
- Entrenamiento lento
```

## Visualizaciones Generadas

### Por Experimento

1. **Training History**
   - Curvas de pérdida (entrenamiento y validación)
   - Curvas de precisión
   - Tiempo por época

2. **Confusion Matrix**
   - Matriz absoluta (counts)
   - Matriz normalizada (%)
   - Identificación de pares confundidos

3. **Decision Boundary**
   - Proyección PCA a 2D
   - Regiones de decisión coloreadas
   - Muestras superpuestas

4. **Loss Landscape**
   - Superficie 3D de pérdida
   - Contornos 2D
   - Punto actual marcado

5. **Weight Distributions**
   - Histogramas por capa
   - Estadísticas (media, std)
   - Evolución durante entrenamiento

## Formato de Reportes

### Reporte Individual (por modelo)

```
1. ARQUITECTURA
   - Detalles de capas
   - Total de parámetros

2. ESTADÍSTICAS DE PESOS
   - Por capa: media, std, min, max

3. DINÁMICA DE ENTRENAMIENTO
   - Reducción de pérdida
   - Mejora de precisión
   - Tiempo de entrenamiento

4. MÉTRICAS DE RENDIMIENTO
   - Precisión train/test
   - Gap de generalización
   - Estado (overfitting/underfitting)

5. MÉTRICAS DETALLADAS
   - Precision, Recall, F1 por clase
   - Matriz de confusión
   - Pares más confundidos

6. ANÁLISIS ESTADÍSTICO
   - Confianza promedio
   - Diferencia de confianza (correctos vs incorrectos)

7. ANÁLISIS DE CONVERGENCIA
   - Gradiente de pérdida
   - Estado de convergencia
```

### Reporte Comparativo

```
1. ESTADÍSTICAS RESUMIDAS
   - Mejor/peor/promedio por métrica

2. TOP 5 CONFIGURACIONES
   - Detalles de cada una
   - Métricas clave

3. ANÁLISIS DE TRADE-OFFS
   - Precisión vs tiempo
   - Complejidad vs rendimiento
```

## Conclusiones y Recomendaciones

(Se completará después de ejecutar experimentos)

### Configuración Recomendada

```
Para uso general en MNIST:
- Capas: [128, 64]
- LR: 0.01
- Activation: sigmoid o tanh
- Epochs: 50-100
- Batch size: 64
```

### Lecciones Aprendidas

1. **Arquitectura**
   - ...

2. **Hiperparámetros**
   - ...

3. **Robustez**
   - ...

## Referencias

- LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
- He, K., et al. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.
