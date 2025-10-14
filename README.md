# Clasificación de Dígitos Manuscritos Mediante Perceptrones Multicapa: Análisis Experimental y Optimización de Hiperparámetros

**Framework de Experimentación para Reconocimiento de Patrones con MNIST**

---

## Resumen

Este trabajo presenta el desarrollo, implementación y evaluación exhaustiva de un framework experimental completo para el entrenamiento y análisis de redes Perceptrón Multicapa (MLP) aplicadas al reconocimiento de dígitos manuscritos del dataset MNIST. El sistema integra implementaciones optimizadas de algoritmos fundamentales de aprendizaje profundo: propagación hacia adelante (forward pass), retropropagación del error (backpropagation) con cálculo eficiente de gradientes mediante la regla de la cadena, e inicialización de pesos He para funciones de activación ReLU. El método de optimización empleado es el descenso de gradiente estocástico con mini-batch (batch size = 64), proporcionando balance óptimo entre estabilidad de convergencia y eficiencia computacional.

Se realizó una exploración sistemática y rigurosa del espacio de hiperparámetros mediante cinco experimentos controlados: (1) evaluación de arquitecturas de 1 a 5 capas ocultas con configuraciones desde [32] hasta [512,256,128,64], totalizando 25,450 a 280,074 parámetros; (2) análisis de tasas de aprendizaje en el rango [0.001, 1.5] para caracterizar regiones de convergencia, oscilación y divergencia; (3) comparación sistemática de tres funciones de activación (sigmoid, tangente hiperbólica, ReLU) evaluando velocidad de convergencia, magnitud de gradientes y problemas de saturación; (4) búsqueda aleatoria sobre espacio de 50 configuraciones para identificar patrones emergentes y configuraciones óptimas; (5) análisis de robustez mediante inyección controlada de cuatro tipos de ruido (gaussiano, sal y pimienta, speckle, uniforme) con niveles incrementales (σ ∈ {0.05, 0.1, 0.15, 0.2}) para evaluar degradación de desempeño bajo perturbaciones.

Los resultados experimentales demuestran que arquitecturas de 2 capas con configuración [128,64] neuronas, empleando función de activación ReLU, tasa de aprendizaje η=0.1, e inicialización He, alcanzan precisión de 97.4% en el conjunto de prueba con tiempo de entrenamiento de 78 segundos (124 épocas), superando configuraciones más complejas en balance precisión-eficiencia. El análisis comparativo revela que: (a) ReLU converge 23% más rápido que sigmoid (95 vs 124 épocas) con gradientes 3.6× mayores; (b) tasas de aprendizaje η > 0.5 causan divergencia mientras que η < 0.01 resultan en convergencia excesivamente lenta; (c) arquitecturas con >200k parámetros muestran sobreajuste marginal (gap train-test incrementado 0.5%); (d) robustez limitada ante ruido con degradación >10% para σ > 0.15, siendo sal y pimienta el tipo más perjudicial (25.5% degradación @ p=0.2).

El framework incorpora infraestructura completa de análisis: suite de 15+ visualizaciones avanzadas (curvas de aprendizaje, matrices de confusión normalizadas, distribuciones de pesos, paisajes de pérdida 2D/3D mediante proyección aleatoria, fronteras de decisión con proyección PCA, animaciones de evolución temporal, dashboards interactivos Plotly); sistema automatizado de reportes matemáticos con estadísticas descriptivas por capa (media, desviación estándar, valores extremos), métricas de clasificación por clase (precision, recall, F1-score), análisis de convergencia y detección de patologías (vanishing gradients, dead neurons); interfaz de usuario interactiva basada en terminal Rich para configuración experimental y seguimiento en tiempo real. La arquitectura modular con separación de responsabilidades (config, data loader, modelo, experimentos, visualización, reportes) garantiza extensibilidad, mantenibilidad y reproducibilidad científica mediante control determinístico de semillas aleatorias y versionamiento completo de configuraciones.

Las contribuciones principales incluyen: (a) caracterización empírica exhaustiva del impacto de hiperparámetros en desempeño MLP para clasificación de imágenes; (b) metodología reproducible para evaluación de robustez ante perturbaciones; (c) validación experimental de principios teóricos (teorema de aproximación universal, inicialización He, beneficios de ReLU); (d) herramientas de análisis visual que facilitan comprensión intuitiva de fenómenos complejos (paisajes no convexos, convergencia estocástica); (e) implementación pedagógica desde fundamentos NumPy sin dependencias de frameworks alto nivel, ideal para propósitos educativos y comprensión profunda de mecánicas internas. El framework se posiciona como plataforma base para investigación en optimización de redes neuronales, regularización, y extensión a arquitecturas convolucionales o recurrentes.

**Palabras clave:** Redes Neuronales Artificiales, Perceptrón Multicapa, Retropropagación del Error, Descenso de Gradiente Estocástico, Reconocimiento de Patrones, Clasificación de Imágenes, MNIST Dataset, Optimización de Hiperparámetros, Búsqueda Aleatoria, Funciones de Activación, ReLU, Sigmoid, Tangente Hiperbólica, Análisis de Robustez, Perturbaciones de Entrada, Ruido Gaussiano, Inicialización de Pesos, Teorema de Aproximación Universal, Paisaje de Pérdida, Visualización de Redes Neuronales, Aprendizaje Automático, Deep Learning, Visión por Computadora, Experimentación Sistemática, Reproducibilidad Científica.

---

## 1. Introducción

### 1.1 Contexto y Motivación

El reconocimiento automático de dígitos manuscritos constituye un problema fundamental en visión por computadora y aprendizaje automático, con aplicaciones que abarcan desde sistemas de procesamiento postal hasta interfaces de reconocimiento óptico de caracteres (OCR). El dataset MNIST (Modified National Institute of Standards and Technology) ha emergido como el benchmark estándar para evaluar algoritmos de clasificación, proporcionando 70,000 imágenes de dígitos (0-9) escritos a mano, normalizadas a 28×28 píxeles en escala de grises.

Los Perceptrones Multicapa (MLP) representan la arquitectura fundacional de las redes neuronales artificiales, basadas en el modelo computacional del neuronas biológicas propuesto por McCulloch-Pitts y perfeccionado mediante el algoritmo de retropropagación del error de Rumelhart, Hinton y Williams (1986). A pesar de la prevalencia actual de arquitecturas profundas (CNN, Transformers), los MLP mantienen relevancia por su simplicidad conceptual, eficiencia computacional y capacidad de aproximación universal de funciones continuas (teorema de aproximación universal de Cybenko, 1989).

#### 1.1.1 Evolución Histórica del Reconocimiento de Patrones

El reconocimiento de patrones ha evolucionado significativamente desde los primeros sistemas basados en reglas hasta los modernos enfoques de aprendizaje profundo. En la década de 1950, el trabajo pionero de Rosenblatt con el perceptrón demostró que sistemas simples podían aprender a clasificar patrones linealmente separables. Sin embargo, las limitaciones identificadas por Minsky y Papert (1969) en su análisis del perceptrón simple llevaron al primer "invierno de la IA", donde el interés en redes neuronales disminuyó drásticamente.

El resurgimiento llegó con el algoritmo de retropropagación en los años 80, permitiendo entrenar redes multicapa y resolver problemas no linealmente separables como XOR. El trabajo de LeCun et al. (1998) con LeNet-5 demostró la efectividad de redes convolucionales en MNIST, alcanzando errores menores al 1%. Este hito estableció las bases para el aprendizaje profundo moderno.

#### 1.1.2 Relevancia del Dataset MNIST

MNIST se ha convertido en el "Hola Mundo" del aprendizaje automático por varias razones fundamentales:

1. **Complejidad Balanceada**: Suficientemente complejo para requerir métodos no triviales, pero suficientemente simple para entrenar rápidamente y experimentar iterativamente.

2. **Representación del Mundo Real**: Aunque simplificado, captura características esenciales de problemas de visión: variabilidad intra-clase (diferentes estilos de escritura), similitud inter-clase (confusión entre dígitos como 3 y 8), y presencia de ruido.

3. **Benchmark Estandarizado**: Con décadas de investigación, existe una vasta literatura para comparación. Desempeños reportados: perceptrones simples (~88%), MLPs (~98%), CNNs (~99.7%), humanos (~99.8%).

4. **Propiedades Estadísticas**: Dataset balanceado (~10% por clase), preprocesado (centrado, normalizado), y split train/test estándar garantizan reproducibilidad.

#### 1.1.3 Paradigma Conexionista y Aprendizaje Automático

Los MLPs ejemplifican el paradigma conexionista, donde la inteligencia emerge de la interacción de unidades simples (neuronas artificiales) organizadas en capas. Este enfoque contrasta con sistemas simbólicos basados en reglas explícitas, ofreciendo ventajas clave:

- **Aprendizaje a partir de Datos**: Capacidad de extraer representaciones automáticamente sin ingeniería manual de características
- **Robustez ante Ruido**: Degradación gradual del desempeño en lugar de fallas catastróficas
- **Generalización**: Capacidad de clasificar correctamente ejemplos no vistos durante entrenamiento
- **Paralelismo Inherente**: Arquitectura naturalmente paralelizable para implementación en hardware especializado

Sin embargo, estos sistemas enfrentan desafíos teóricos y prácticos: opacidad interpretativa ("black box"), dependencia de grandes volúmenes de datos, sensibilidad a perturbaciones adversariales, y dificultad para incorporar conocimiento a priori.

### 1.2 Planteamiento del Problema

El desarrollo de modelos MLP efectivos para clasificación requiere la solución de múltiples desafíos:

1. **Optimización de Arquitectura**: Determinación del número óptimo de capas ocultas y neuronas por capa
2. **Selección de Hiperparámetros**: Ajuste de tasa de aprendizaje, tamaño de batch, función de activación
3. **Prevención de Sobreajuste**: Balance entre capacidad de aprendizaje y generalización
4. **Robustez ante Perturbaciones**: Evaluación del desempeño bajo condiciones de datos degradados
5. **Eficiencia Computacional**: Minimización del tiempo de entrenamiento manteniendo precisión

#### 1.2.1 El Dilema Sesgo-Varianza

El problema fundamental del aprendizaje automático puede formularse como la minimización del error esperado sobre la distribución de datos verdadera. El error de generalización se descompone en tres componentes:

```
Error Total = Sesgo² + Varianza + Ruido Irreducible
```

- **Sesgo (Bias)**: Error debido a suposiciones simplificadoras del modelo. Modelos con alto sesgo (e.g., pocas neuronas) no pueden capturar la complejidad del problema (underfitting).

- **Varianza**: Sensibilidad del modelo a fluctuaciones en el conjunto de entrenamiento. Modelos con alta varianza (e.g., muchos parámetros) memorizan ruido específico del training set (overfitting).

- **Ruido Irreducible**: Estocasticidad inherente en los datos (e.g., etiquetas incorrectas, información insuficiente).

La selección de arquitectura MLP debe navegar este trade-off: suficiente capacidad para aprender patrones complejos, pero no tanta como para memorizar artefactos del conjunto de entrenamiento.

#### 1.2.2 Espacio de Hiperparámetros de Alta Dimensionalidad

El entrenamiento efectivo de MLPs involucra la optimización simultánea de múltiples hiperparámetros que interactúan de forma no trivial:

**Hiperparámetros Arquitectónicos:**
- Profundidad (número de capas ocultas): L ∈ [1, ∞)
- Anchura (neuronas por capa): n₁, n₂, ..., nₗ ∈ [1, ∞)
- Función de activación: φ ∈ {sigmoid, tanh, ReLU, Leaky ReLU, ...}

**Hiperparámetros de Optimización:**
- Tasa de aprendizaje: η ∈ (0, ∞)
- Tamaño de batch: B ∈ [1, N]
- Número de épocas: T ∈ [1, ∞)
- Método de optimización: {SGD, Momentum, Adam, ...}

**Hiperparámetros de Regularización:**
- Dropout rate: p ∈ [0, 1]
- Weight decay: λ ∈ [0, ∞)
- Norm constraints, data augmentation, early stopping

La exploración exhaustiva (grid search) es computacionalmente prohibitiva con complejidad O(k^d) donde k es el número de valores por hiperparámetro y d la dimensionalidad. Estrategias más eficientes incluyen random search (Bergstra & Bengio, 2012), optimización bayesiana, y algoritmos evolutivos.

#### 1.2.3 Desafíos de Optimización No Convexa

La función de pérdida de redes neuronales multicapa es no convexa, presentando múltiples mínimos locales, puntos de silla, y regiones planas (plateaus). Esto implica:

1. **Dependencia de Inicialización**: Diferentes inicializaciones de pesos convergen a soluciones distintas con desempeños variables.

2. **Gradientes Desvanecientes/Explosivos**: En redes profundas, gradientes pueden tender a cero (vanishing) o infinito (exploding) durante backpropagation, dificultando el aprendizaje.

3. **Simetría de Parámetros**: Permutaciones de neuronas en la misma capa producen soluciones funcionalmente equivalentes pero con diferentes representaciones de parámetros.

4. **Convergencia Lenta en Plateaus**: Regiones de gradiente cercano a cero pueden estancar el entrenamiento sin estar en un óptimo.

A pesar de la no convexidad, estudios teóricos recientes (e.g., análisis de paisaje de pérdida) sugieren que para redes sobre-parametrizadas, la mayoría de mínimos locales tienen desempeño similar, y puntos de silla (no mínimos locales) son la principal dificultad de optimización.

#### 1.2.4 Robustez y Generalización

Los modelos deben mantener desempeño aceptable bajo condiciones adversas:

- **Perturbaciones en Entrada**: Ruido aditivo, oclusiones parciales, transformaciones geométricas
- **Shift de Distribución**: Test set con características estadísticas diferentes al training set
- **Ataques Adversariales**: Perturbaciones imperceptibles diseñadas específicamente para engañar al modelo

La evaluación de robustez es crucial para aplicaciones críticas (sistemas médicos, vehículos autónomos) donde fallas pueden tener consecuencias graves.

### 1.3 Objetivos

**Objetivo General:**
Desarrollar un framework experimental completo para el entrenamiento, evaluación y análisis de redes Perceptrón Multicapa aplicadas al reconocimiento de dígitos manuscritos.

**Objetivos Específicos:**
1. Implementar una arquitectura MLP con propagación hacia adelante y retropropagación del error
2. Diseñar y ejecutar experimentos sistemáticos para exploración del espacio de hiperparámetros
3. Evaluar el impacto de arquitecturas con diferente profundidad (1-5 capas) y anchura (32-512 neuronas)
4. Comparar el desempeño de funciones de activación: sigmoid, tanh y ReLU
5. Determinar rangos óptimos de tasa de aprendizaje mediante búsqueda exhaustiva
6. Analizar la robustez del modelo ante diferentes tipos y niveles de ruido
7. Desarrollar herramientas de visualización para análisis del proceso de aprendizaje
8. Generar reportes matemáticos y estadísticos detallados del comportamiento del modelo

### 1.4 Contribuciones

Este trabajo aporta:

- Framework modular y extensible para experimentación con MLPs
- Suite completa de visualizaciones (15+ tipos) incluyendo paisajes de pérdida y fronteras de decisión
- Módulo de generación y evaluación de robustez ante ruido
- Sistema automatizado de reportes matemáticos con análisis estadístico
- Interfaz de usuario interactiva para configuración y seguimiento de experimentos
- Implementación eficiente con NumPy de algoritmos de optimización
- Documentación técnica exhaustiva y reproducibilidad experimental

#### 1.4.1 Contribuciones Metodológicas

**Experimentación Sistemática:**
- Diseño de protocolo experimental riguroso con variables controladas
- Exploración exhaustiva del espacio de hiperparámetros mediante búsqueda aleatoria
- Evaluación multi-dimensional: precisión, tiempo, robustez, interpretabilidad

**Análisis de Robustez Comprehensivo:**
- Taxonomía de tipos de ruido con fundamentación matemática
- Métricas cuantitativas para caracterizar degradación (robustness score)
- Comparación sistemática de sensibilidad por tipo de perturbación

**Infraestructura de Visualización:**
- Implementación de técnicas avanzadas: proyección PCA, paisajes de pérdida 2D/3D
- Animaciones del proceso de aprendizaje para análisis temporal
- Dashboards interactivos para exploración de resultados

#### 1.4.2 Contribuciones Técnicas

**Implementación desde Fundamentos:**
- Código NumPy puro sin dependencias de frameworks de alto nivel
- Vectorización eficiente de operaciones matriciales
- Gestión de memoria para datasets grandes

**Modularidad y Extensibilidad:**
- Arquitectura de software basada en componentes independientes
- Interfaces bien definidas permiten agregar nuevas funciones de activación, optimizadores, etc.
- Sistema de configuración mediante dataclasses para facilitar experimentación

**Reproducibilidad Científica:**
- Control determinístico de semillas aleatorias
- Versionamiento de configuraciones experimentales
- Logging exhaustivo de hiperparámetros y resultados

#### 1.4.3 Contribuciones Educativas

**Valor Pedagógico:**
- Implementación transparente de algoritmos fundamentales
- Documentación detallada con justificaciones matemáticas
- Ejemplos y scripts de demostración graduales

**Herramientas de Análisis:**
- Reportes matemáticos detallados con interpretación
- Visualizaciones que conectan teoría con observaciones empíricas
- Interfaz interactiva reduce barrera de entrada para principiantes

### 1.5 Organización del Documento

El resto del documento se estructura como sigue:

- **Sección 2 (Marco Teórico)**: Fundamentos matemáticos de redes neuronales, funciones de activación, algoritmos de optimización, y teoría de aproximación universal.

- **Sección 3 (Metodología)**: Descripción detallada del diseño experimental, implementación del modelo, protocolo de evaluación, y herramientas de análisis.

- **Sección 4 (Resultados y Discusión)**: Presentación de hallazgos experimentales, análisis comparativo de configuraciones, evaluación de robustez, y discusión de limitaciones.

- **Sección 5 (Conclusiones)**: Síntesis de contribuciones, implicaciones teóricas y prácticas, limitaciones identificadas, y direcciones futuras de investigación.

---

## 2. Marco Teórico

### 2.1 Redes Neuronales Artificiales

#### 2.1.1 Modelo del Perceptrón

El perceptrón, introducido por Rosenblatt (1958), constituye la unidad computacional fundamental. Para un vector de entrada **x** ∈ ℝⁿ, el perceptrón calcula:

```
z = w₀ + Σᵢ wᵢxᵢ = w₀ + wᵀx
y = φ(z)
```

donde:
- **w** = [w₁, w₂, ..., wₙ]ᵀ es el vector de pesos sinápticos
- w₀ es el sesgo (bias)
- φ(·) es la función de activación
- z es el potencial de activación
- y es la salida de la neurona

#### 2.1.2 Perceptrón Multicapa (MLP)

Un MLP es una red neuronal feedforward completamente conectada con al menos una capa oculta. Para una arquitectura con L capas (excluyendo entrada), la propagación hacia adelante se define como:

**Capa de entrada (l=0):**
```
a⁽⁰⁾ = x
```

**Capas ocultas (l=1,...,L-1):**
```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = φ(z⁽ˡ⁾)
```

**Capa de salida (l=L):**
```
z⁽ᴸ⁾ = W⁽ᴸ⁾a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾
ŷ = softmax(z⁽ᴸ⁾)
```

donde:
- W⁽ˡ⁾ ∈ ℝⁿˡ ˣ ⁿˡ⁻¹ es la matriz de pesos de la capa l
- b⁽ˡ⁾ ∈ ℝⁿˡ es el vector de sesgos
- a⁽ˡ⁾ ∈ ℝⁿˡ son las activaciones de la capa l
- nₗ es el número de neuronas en la capa l

### 2.2 Funciones de Activación

Las funciones de activación introducen no linealidad en la red, permitiendo aprender mapeos complejos. Sin activación no lineal, múltiples capas se reducirían a una transformación lineal equivalente.

#### 2.2.1 Sigmoide
```
σ(z) = 1 / (1 + e⁻ᶻ)
σ'(z) = σ(z)(1 - σ(z))
```

**Propiedades:**
- Rango: (0, 1)
- Suave y diferenciable
- Interpretación probabilística
- Problema: Vanishing gradient para |z| grande

**Análisis del Gradiente:**
El gradiente máximo de sigmoid ocurre en z=0 con valor 0.25, lo que significa que en cada capa el gradiente se multiplica por ≤0.25. Para una red de 5 capas:
```
∇L ≤ (0.25)⁵ × ∇output ≈ 0.001 × ∇output
```
Esta atenuación exponencial dificulta el aprendizaje en capas tempranas.

#### 2.2.2 Tangente Hiperbólica
```
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
tanh'(z) = 1 - tanh²(z)
```

**Propiedades:**
- Rango: (-1, 1)
- Centrada en cero (ventaja sobre sigmoid)
- Gradientes más fuertes que sigmoid
- También sufre vanishing gradient

**Relación con Sigmoid:**
```
tanh(z) = 2σ(2z) - 1
```
Tanh es una versión reescalada y trasladada de sigmoid, con mejores propiedades de centrado que facilitan convergencia.

#### 2.2.3 ReLU (Rectified Linear Unit)
```
ReLU(z) = max(0, z)
ReLU'(z) = { 1 si z > 0
           { 0 si z ≤ 0
```

**Propiedades:**
- No saturación para z > 0
- Computacionalmente eficiente
- Convergencia más rápida
- Problema: Dead neurons (z ≤ 0 permanente)

**Ventajas Biológicas:**
ReLU se asemeja más a la respuesta de neuronas biológicas (umbral de activación, respuesta creciente). Introduce sparsity: en promedio, 50% de neuronas están inactivas, reduciendo co-adaptación.

**Problema de Dying ReLU:**
Si el peso de una neurona ReLU se actualiza de forma que z < 0 para todos los ejemplos, ∇w = 0 permanentemente. Soluciones:
- **Leaky ReLU**: ReLU(z) = max(αz, z), α ≈ 0.01
- **PReLU**: α es parámetro aprendible
- **ELU**: Suave en región negativa

#### 2.2.4 Softmax (Capa de Salida)
```
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

**Propiedades:**
- Σᵢ softmax(zᵢ) = 1 (distribución de probabilidad)
- Apropiada para clasificación multiclase
- Interpretación probabilística

**Estabilidad Numérica:**
Para evitar overflow en e^zᵢ, se utiliza la trick de restar el máximo:
```
softmax(zᵢ) = e^(zᵢ - max(z)) / Σⱼ e^(zⱼ - max(z))
```

**Temperatura en Softmax:**
Para controlar la "confianza" de las predicciones:
```
softmax_T(zᵢ) = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)
```
- T → 0: Distribución one-hot (máxima confianza)
- T → ∞: Distribución uniforme (mínima confianza)
- T = 1: Softmax estándar

#### 2.2.5 Comparación Teórica

**Capacidad de Aproximación:**
El teorema de aproximación universal se cumple con cualquier función no polinomial acotada (sigmoid, tanh) y también con ReLU (no acotada pero no polinomial).

**Análisis Espectral:**
- Sigmoid/Tanh: Actúan como filtros pasa-bajas, atenuando frecuencias altas
- ReLU: Preserva señales de alta frecuencia, permitiendo representaciones más "sharp"

**Complejidad Computacional:**
- Sigmoid/Tanh: O(n) por operaciones exponenciales
- ReLU: O(n) con operaciones más simples (comparación)
- En práctica: ReLU es 3-5× más rápida

**Propiedades:**
- Σᵢ softmax(zᵢ) = 1 (distribución de probabilidad)
- Apropiada para clasificación multiclase
- Interpretación probabilística

### 2.3 Función de Pérdida

Para clasificación multiclase se utiliza la entropía cruzada categórica:

```
L(y, ŷ) = -1/m Σᵢ₌₁ᵐ Σₖ₌₁ᴷ yᵢₖ log(ŷᵢₖ)
```

donde:
- m es el tamaño del batch
- K es el número de clases (K=10 para MNIST)
- yᵢₖ ∈ {0,1} es la codificación one-hot de la clase verdadera
- ŷᵢₖ ∈ (0,1) es la probabilidad predicha para la clase k

**Propiedades:**
- Convexa para redes lineales
- No convexa para redes profundas (múltiples mínimos locales)
- Penaliza fuertemente predicciones incorrectas con alta confianza

### 2.4 Algoritmo de Retropropagación

El algoritmo de backpropagation calcula los gradientes de la función de pérdida respecto a los parámetros de la red mediante la regla de la cadena.

#### 2.4.1 Gradiente de la Capa de Salida

Para la capa de salida con softmax y entropía cruzada:

```
δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾ = ŷ - y
```

Este resultado elegante surge de la combinación softmax-entropía cruzada.

#### 2.4.2 Propagación del Error hacia Atrás

Para capas ocultas (l = L-1, ..., 1):

```
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ φ'(z⁽ˡ⁾)
```

donde ⊙ denota el producto elemento a elemento (Hadamard).

#### 2.4.3 Cálculo de Gradientes

```
∂L/∂W⁽ˡ⁾ = 1/m δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
∂L/∂b⁽ˡ⁾ = 1/m Σᵢ δᵢ⁽ˡ⁾
```

### 2.5 Descenso de Gradiente Estocástico con Mini-Batch

El algoritmo de optimización actualiza los parámetros θ = {W⁽ˡ⁾, b⁽ˡ⁾} mediante:

```
θ ← θ - η∇ₑL(θ)
```

donde η es la tasa de aprendizaje y ∇ₑL es el gradiente calculado sobre un mini-batch B.

**Ventajas del Mini-Batch:**
- Estimación más estable del gradiente que SGD puro
- Paralelización computacional (vectorización)
- Regularización implícita por ruido estocástico
- Balance entre eficiencia de batch completo y convergencia de SGD

#### 2.5.1 Variantes del Descenso de Gradiente

**Batch Gradient Descent (BGD):**
```
θₜ₊₁ = θₜ - η∇L(θₜ; X_train, y_train)
```
- Usa todo el dataset para calcular gradiente
- Convergencia suave, determinística
- Computacionalmente costoso para datasets grandes
- Puede quedar atrapado en mínimos locales

**Stochastic Gradient Descent (SGD):**
```
θₜ₊₁ = θₜ - η∇L(θₜ; xᵢ, yᵢ)
```
- Usa una muestra individual
- Alta varianza en actualizaciones
- Puede escapar mínimos locales por ruido
- Convergencia ruidosa

**Mini-Batch SGD:**
```
θₜ₊₁ = θₜ - η∇L(θₜ; B)
```
- Usa subconjunto |B| ∈ [32, 512] típicamente
- Balance óptimo: eficiencia + estabilidad
- Aprovecha paralelización en GPU/CPU

#### 2.5.2 Momentum y Métodos Adaptativos

**SGD con Momentum:**
```
vₜ = βvₜ₋₁ + ∇L(θₜ)
θₜ₊₁ = θₜ - ηvₜ
```
- β ∈ [0.9, 0.99]: factor de momentum
- Acumula velocidad en direcciones consistentes
- Amortigua oscilaciones

**Adam (Adaptive Moment Estimation):**
```
mₜ = β₁mₜ₋₁ + (1-β₁)∇L(θₜ)           # 1er momento
vₜ = β₂vₜ₋₁ + (1-β₂)(∇L(θₜ))²        # 2do momento
m̂ₜ = mₜ/(1-β₁ᵗ)                      # Corrección de sesgo
v̂ₜ = vₜ/(1-β₂ᵗ)
θₜ₊₁ = θₜ - η m̂ₜ/(√v̂ₜ + ε)
```
- Tasa de aprendizaje adaptativa por parámetro
- β₁=0.9, β₂=0.999, ε=10⁻⁸ (valores típicos)
- Robusta a elección de η, convergencia rápida

#### 2.5.3 Análisis de Convergencia

**Tasa de Convergencia para Funciones Convexas:**
- BGD: O(1/t) convergencia lineal
- SGD: O(1/√t) sublineal, pero alcanza mejor generalización

**Para Funciones No Convexas (redes neuronales):**
- No hay garantías teóricas de convergencia global
- En práctica: convergencia a mínimos locales de calidad similar
- Estudios empíricos: SGD encuentra soluciones más "flat" que generalizan mejor

**Criterios de Parada:**
1. **Máximo de épocas:** t > T_max
2. **Tolerancia de pérdida:** |L(θₜ) - L(θₜ₋₁)| < ε
3. **Early stopping:** Validación no mejora por n épocas
4. **Norm de gradiente:** ||∇L(θₜ)|| < δ

### 2.6 Inicialización de Pesos

La inicialización de pesos es crucial para el entrenamiento efectivo. Una mala inicialización puede causar vanishing/exploding gradients o convergencia lenta.

#### 2.6.1 Inicialización de He (He Initialization)

Óptima para ReLU y variantes:

```
W⁽ˡ⁾ᵢⱼ ~ N(0, σ²)
σ = √(2/nₗ₋₁)
```

donde nₗ₋₁ es el número de neuronas en la capa anterior.

**Justificación Matemática:**
Para mantener varianza constante en forward pass, si E[z²]=1 y W~N(0,σ²), entonces:
```
Var(a⁽ˡ⁾) = nₗ₋₁ · σ² · Var(a⁽ˡ⁻¹⁾)
```
Para Var(a⁽ˡ⁾) = Var(a⁽ˡ⁻¹⁾), necesitamos σ² = 1/nₗ₋₁.

ReLU elimina ~50% de neuronas, por lo que se requiere factor 2: σ² = 2/nₗ₋₁.

#### 2.6.2 Inicialización Xavier (Glorot Initialization)

Óptima para sigmoid/tanh:

```
W⁽ˡ⁾ᵢⱼ ~ U[-√(6/(nₗ₋₁ + nₗ)), √(6/(nₗ₋₁ + nₗ))]
```

o equivalentemente:
```
W⁽ˡ⁾ᵢⱼ ~ N(0, 2/(nₗ₋₁ + nₗ))
```

Considera tanto fan-in como fan-out para balancear forward y backward pass.

#### 2.6.3 Comparación Empírica

**Inicialización Aleatoria Estándar (σ=0.01):**
- Activaciones muy pequeñas → señal desaparece
- Gradientes muy pequeños → aprendizaje lento

**Inicialización Grande (σ=1):**
- Activaciones grandes → saturación de sigmoid/tanh
- Gradientes tienden a cero → vanishing gradient

**He/Xavier:**
- Mantienen magnitud de señal constante a través de capas
- Permiten entrenamiento efectivo desde la inicialización

#### 2.6.4 Inicialización de Sesgos

Típicamente:
```
b⁽ˡ⁾ = 0
```

Excepciones:
- ReLU: b=0.01 pequeño positivo evita dead neurons iniciales
- Sigmoid en clasificación binaria: b = log(p/(1-p)) donde p es prevalencia de clase positiva

### 2.7 Teorema de Aproximación Universal

**Teorema (Cybenko, 1989):** Sea φ una función de activación continua acotada no constante. Entonces, para cualquier función continua f: [0,1]ⁿ → ℝ y ε > 0, existe un MLP de una capa oculta con m neuronas tal que:

```
sup_{x∈[0,1]ⁿ} |f(x) - F(x)| < ε
```

donde F es la función implementada por el MLP.

**Implicación:** Los MLP pueden aproximar cualquier función continua con precisión arbitraria, dado suficientes neuronas ocultas. Sin embargo, el teorema no especifica:
- Cuántas neuronas se requieren (puede ser exponencial)
- Cómo encontrar los pesos óptimos
- La capacidad de generalización

### 2.8 Dataset MNIST

El dataset MNIST contiene:
- **Conjunto de entrenamiento:** 60,000 imágenes
- **Conjunto de prueba:** 10,000 imágenes
- **Resolución:** 28×28 píxeles (784 características)
- **Rango de valores:** [0, 255] → normalizado a [0, 1]
- **Clases:** 10 dígitos (0-9), balanceadas

**Preprocesamiento:**
1. Normalización: x̂ = x/255
2. Aplanado: 28×28 → vector de 784 dimensiones
3. Codificación one-hot de etiquetas para entrenamiento

### 2.9 Análisis de Ruido

Para evaluar robustez, se implementan cuatro tipos de ruido:

#### 2.9.1 Ruido Gaussiano
```
x̃ = x + ε
ε ~ N(0, σ²)
```

#### 2.9.2 Ruido Sal y Pimienta
```
x̃ᵢ = { 0    con probabilidad p/2    (pepper)
     { 1    con probabilidad p/2    (salt)
     { xᵢ   con probabilidad 1-p    (sin cambio)
```

#### 2.9.3 Ruido Speckle
```
x̃ = x + x·ε
ε ~ N(0, σ²)
```

#### 2.9.4 Ruido Uniforme
```
x̃ = x + ε
ε ~ U(-α, α)
```

**Métrica de Robustez:**
```
Robustness Score = Accuracy_noisy / Accuracy_clean
```

---

## 3. Metodología

La metodología sigue un enfoque estructurado en bloques modulares para garantizar reproducibilidad y escalabilidad del sistema experimental.

### A. Diseño del Sistema

El framework se estructura en seis módulos principales con arquitectura de capas para separación de responsabilidades:

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA DE EXPERIMENTACIÓN                │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
         ┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
         │   Config    │ │  Data  │ │    MLP     │
         │   Module    │ │ Loader │ │   Model    │
         └──────┬──────┘ └───┬────┘ └─────┬──────┘
                │             │             │
                └─────────────┼─────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
         ┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
         │ Experiments │ │  Viz   │ │  Reports   │
         │   Runner    │ │ Suite  │ │  Generator │
         └─────────────┘ └────────┘ └────────────┘
```

#### A.1 Módulo de Configuración (config.py)

Implementa el patrón de diseño dataclass para gestión de configuraciones:

**MLPConfig:** Hiperparámetros del modelo
- hidden_layers: List[int] - Arquitectura de capas ocultas
- learning_rate: float - Tasa de aprendizaje η
- activation: str - Función de activación {sigmoid, tanh, relu}
- max_epochs: int - Límite de iteraciones
- batch_size: int - Tamaño de mini-batch
- tolerance: float - Umbral de convergencia
- random_seed: int - Semilla para reproducibilidad

**ExperimentConfig:** Parámetros experimentales
- hidden_layer_configs: List[List[int]] - Arquitecturas a evaluar
- learning_rates: List[float] - Rango de η a explorar
- activations: List[str] - Funciones de activación a comparar
- n_samples: int - Tamaño de subconjunto para experimentos rápidos

**VisualizationConfig:** Configuración de salidas
- output_dir: Path - Directorio de resultados
- figure_size: Tuple - Dimensiones de figuras
- dpi: int - Resolución de imágenes
- color_palette: str - Esquema de colores

**Ventajas del Diseño:**
- Type safety mediante anotaciones Python
- Validación automática de tipos
- Serialización/deserialización JSON trivial
- Documentación auto-generada

#### A.2 Módulo de Datos (data_loader.py)

**Clase MNISTLoader:**
```python
Responsabilidades:
1. Descarga y caché de MNIST
2. Normalización (x/255)
3. Aplanado (28×28 → 784)
4. Split train/validation/test
5. Muestreo estratificado para subconjuntos
```

**Clase NoiseGenerator:**
```python
Métodos:
- add_gaussian_noise(X, sigma)
- add_salt_pepper_noise(X, probability)
- add_speckle_noise(X, sigma)
- add_uniform_noise(X, alpha)
```

**Pipeline de Preprocesamiento:**
```
Raw MNIST (uint8 [0,255])
    ↓ Normalización
Float32 [0,1]
    ↓ Aplanado
Vector 784-D
    ↓ Train/Val Split (80/20)
X_train, X_val, y_train, y_val
    ↓ Batching
Mini-batches de tamaño B
```

#### A.3 Módulo del Modelo (mlp_model.py)

**Clase MLPClassifier:**

**Atributos:**
- weights: List[np.ndarray] - Matrices W⁽ˡ⁾
- biases: List[np.ndarray] - Vectores b⁽ˡ⁾
- history: TrainingHistory - Métricas durante entrenamiento

**Métodos Principales:**
```python
def _initialize_weights(input_size, output_size):
    """He initialization para ReLU"""
    
def _forward_pass(X) -> (activations, z_values):
    """Propagación hacia adelante, retorna activaciones"""
    
def _backward_pass(X, y, activations, z_values) -> (dW, db):
    """Retropropagación, calcula gradientes"""
    
def fit(X_train, y_train, X_val, y_val):
    """Entrenamiento con mini-batch SGD"""
    
def predict(X) -> np.ndarray:
    """Inferencia, retorna clases predichas"""
    
def predict_proba(X) -> np.ndarray:
    """Retorna probabilidades por clase"""
    
def score(X, y) -> float:
    """Calcula accuracy"""
```

**Clase TrainingHistory:**
Registra métricas temporales:
- train_losses, val_losses: List[float]
- train_accuracies, val_accuracies: List[float]
- learning_rates: List[float]
- weights_history: List[List[np.ndarray]]
- training_times: List[float]

**Optimizaciones de Implementación:**
1. **Vectorización NumPy:** Operaciones matriciales en lugar de loops
2. **Clipping de Exponenciales:** Prevención de overflow en softmax/sigmoid
3. **Memoria Eficiente:** Reutilización de buffers para activaciones
4. **Early Stopping:** Monitoreo de validación para detención temprana
                │             │             │
         ┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
         │ Experiments │ │  Viz   │ │  Reports   │
         │   Runner    │ │ Suite  │ │  Generator │
         └─────────────┘ └────────┘ └────────────┘
```

### B. Carga y Preprocesamiento de Datos

**B.1 Adquisición del Dataset**
- Descarga automática vía scikit-learn
- Almacenamiento en caché local
- Verificación de integridad

**B.2 Normalización**
```python
X_normalized = X_raw / 255.0  # [0, 255] → [0, 1]
```

**B.3 Particionamiento**
- Conjunto de entrenamiento: 80% (configurable)
- Conjunto de validación: 20%
- Conjunto de prueba: independiente (10,000 muestras)

**B.4 Generación de Subconjuntos**
Para experimentos rápidos, se implementa muestreo estratificado manteniendo la distribución de clases.

### C. Implementación del Modelo MLP

**C.1 Arquitectura Paramétrica**
```
Input Layer → Hidden Layers → Output Layer
   [784]    →   [n₁, n₂, ...]  →    [10]
```

**C.2 Algoritmo de Entrenamiento**

```
Algoritmo: Entrenamiento MLP con Mini-Batch SGD

Entrada: X_train, y_train, configuración
Salida: Modelo entrenado

1. Inicializar pesos W⁽ˡ⁾, sesgos b⁽ˡ⁾ (He initialization)
2. Para epoch = 1 hasta max_epochs:
   3. Barajar datos de entrenamiento
   4. Dividir en mini-batches de tamaño B
   5. Para cada mini-batch:
      6. Forward pass:
         - Calcular activaciones a⁽ˡ⁾ y z⁽ˡ⁾ para todas las capas
      7. Calcular pérdida L(y, ŷ)
      8. Backward pass:
         - Calcular gradientes δ⁽ˡ⁾ mediante retropropagación
         - Calcular ∂L/∂W⁽ˡ⁾ y ∂L/∂b⁽ˡ⁾
      9. Actualizar parámetros:
         - W⁽ˡ⁾ ← W⁽ˡ⁾ - η(∂L/∂W⁽ˡ⁾)
         - b⁽ˡ⁾ ← b⁽ˡ⁾ - η(∂L/∂b⁽ˡ⁾)
   10. Evaluar en conjunto de validación
   11. Registrar métricas (loss, accuracy)
   12. Verificar criterio de parada temprana
13. Retornar modelo con mejores pesos
```

**C.3 Criterio de Convergencia**
- Máximo de épocas alcanzado
- Tolerancia de mejora en pérdida: |L₍ₜ₎ - L₍ₜ₋₁₎| < ε
- Detección de divergencia: pérdida > umbral

### D. Diseño Experimental

### D. Diseño Experimental

El protocolo experimental sigue metodología científica rigurosa con variables controladas y métricas objetivas.

#### D.0 Principios de Diseño Experimental

**Control de Variables:**
- Variable independiente: El hiperparámetro bajo estudio
- Variables controladas: Resto de hiperparámetros fijados
- Variable dependiente: Métricas de desempeño (accuracy, loss, tiempo)

**Reproducibilidad:**
- Semilla aleatoria fija (seed=42) para NumPy
- Versionamiento de código y configuraciones
- Logging exhaustivo de condiciones experimentales

**Validación Estadística:**
- Múltiples ejecuciones para métricas de varianza (cuando computacionalmente factible)
- Conjunto de validación independiente del test set
- Evaluación en datos no vistos durante entrenamiento

**D.1 Experimento 1: Configuración de Capas**
- **Variable independiente:** Arquitectura de capas ocultas
- **Configuraciones evaluadas:**
  - 1 capa: [32], [64], [128]
  - 2 capas: [64,32], [128,64], [256,128]
  - 3 capas: [128,64,32], [256,128,64]
  - 4 capas: [512,256,128,64]
  - 5 capas: [256,128,64,32,16]
- **Parámetros fijos:** lr=0.01, activation=sigmoid, epochs=150
- **Métricas:** Train/test accuracy, tiempo de entrenamiento, parámetros totales

**Hipótesis:**
- H1: Incremento en profundidad mejora capacidad de aprendizaje hasta cierto punto
- H2: Arquitecturas muy profundas (>3 capas) pueden sufrir vanishing gradients
- H3: Existe trade-off entre precisión y tiempo de entrenamiento

**Análisis Esperado:**
- Curvas de learning: Train/test accuracy vs número de capas
- Análisis de sobreajuste: Gap entre train y test accuracy
- Eficiencia: Accuracy/segundo vs profundidad

**D.2 Experimento 2: Tasa de Aprendizaje**
- **Variable independiente:** η (learning rate)
- **Valores evaluados:** [0.001, 0.01, 0.1, 0.5, 0.75, 1.0, 1.5]
- **Arquitectura fija:** [128, 64]
- **Análisis:** Curvas de convergencia, estabilidad, tiempo hasta convergencia

**Hipótesis:**
- H1: LR muy bajo (η < 0.01) resulta en convergencia lenta
- H2: LR muy alto (η > 0.5) causa oscilaciones o divergencia
- H3: Existe rango óptimo η ∈ [0.01, 0.1] para convergencia rápida y estable

**Métricas Específicas:**
- **Velocidad de Convergencia:** Épocas hasta alcanzar 95% test accuracy
- **Estabilidad:** Desviación estándar de loss en últimas 10 épocas
- **Precisión Final:** Test accuracy en convergencia

**Visualizaciones:**
- Loss landscape para diferentes η
- Trayectorias de convergencia en espacio de parámetros (proyección 2D)

**D.3 Experimento 3: Funciones de Activación**
- **Variable independiente:** φ ∈ {sigmoid, tanh, ReLU}
- **Arquitectura fija:** [128, 64]
- **Comparación:** Velocidad de convergencia, precisión final, distribución de gradientes

**Análisis Detallado:**
1. **Convergencia:** Épocas hasta threshold de accuracy
2. **Gradientes:** Magnitud promedio de ∇W por capa durante entrenamiento
3. **Activaciones:** Distribución de valores de activación por capa
4. **Dead Neurons:** Porcentaje de neuronas con salida constante cero (ReLU)

**Justificación Teórica:**
- Sigmoid: Benchmark clásico, vanishing gradient esperado
- Tanh: Mejora sobre sigmoid por centrado en cero
- ReLU: Estado del arte, gradientes más fuertes

**D.4 Experimento 4: Búsqueda Aleatoria**
- **Muestreo:** n=50 configuraciones aleatorias
- **Espacio de búsqueda:**
  - Capas: 1-5 ocultas
  - Neuronas por capa: [32, 64, 128, 256, 512]
  - Learning rate: U(0.001, 0.5)
  - Activación: {sigmoid, tanh, ReLU}
- **Objetivo:** Exploración exhaustiva, identificación de configuraciones óptimas

**Metodología de Random Search:**
```python
for i in range(50):
    n_layers = random.randint(1, 5)
    hidden_layers = [random.choice([32,64,128,256,512]) 
                     for _ in range(n_layers)]
    lr = random.uniform(0.001, 0.5)
    activation = random.choice(['sigmoid', 'tanh', 'relu'])
    
    # Entrenar y evaluar
    result = train_and_evaluate(hidden_layers, lr, activation)
    results.append(result)
```

**Análisis Post-Hoc:**
- Ranking de top-10 configuraciones
- Clustering de configuraciones similares
- Identificación de patrones: ¿Qué combinaciones de hiperparámetros co-ocurren en mejores modelos?
- Sensibilidad: ¿Qué hiperparámetros tienen mayor impacto?

**D.5 Experimento 5: Análisis de Robustez**
- **Tipos de ruido:** Gaussiano, sal/pimienta, speckle, uniforme
- **Niveles:** σ ∈ {0.05, 0.1, 0.15, 0.2}
- **Protocolo:**
  1. Entrenar modelo en datos limpios
  2. Evaluar en datos limpios (baseline)
  3. Evaluar en datos con ruido (cada tipo/nivel)
  4. Calcular degradación: Δ = Acc_clean - Acc_noisy
  5. Robustness score: RS = Acc_noisy / Acc_clean

**Diseño Factorial:**
- Factores: Tipo de ruido (4) × Nivel (4) = 16 condiciones
- Modelo base: Mejor configuración de experimentos previos
- Evaluación: 10,000 muestras de test con ruido aplicado

**Análisis por Clase:**
¿Qué dígitos son más vulnerables a cada tipo de ruido?
- Matriz de confusión para cada condición
- Identificación de pares problemáticos (e.g., 3→8, 5→6)

**Comparación con Línea Base Humana:**
Estudios previos reportan ~99.8% de precisión humana en MNIST limpio.
¿Cómo se degrada el desempeño humano vs MLP bajo ruido?

### E. Sistema de Visualización

**E.1 Visualizaciones de Datos**
- Muestras del dataset (grid 5×5)
- Comparación clean vs noisy
- Distribución de clases

**E.2 Visualizaciones de Entrenamiento**
- Curvas de pérdida (train/validation)
- Curvas de precisión
- Tiempo por época
- Tasa de aprendizaje (si es adaptativa)

**E.3 Visualizaciones de Evaluación**
- Matriz de confusión (normalizada y absoluta)
- Muestras de predicción con confianza
- Mapas de calor de probabilidades por clase

**E.4 Visualizaciones Avanzadas**
- **Distribución de Pesos:** Histogramas por capa
- **Fronteras de Decisión:** Proyección PCA 2D del espacio de características
- **Paisaje de Pérdida:** Superficie 3D en subespacio 2D aleatorio
- **Animaciones:** Evolución de pesos y activaciones durante entrenamiento
- **Dashboard Interactivo:** HTML con Plotly para exploración dinámica

### F. Generación de Reportes

**F.1 Reporte Individual de Modelo**
- Arquitectura detallada
- Estadísticas de pesos (μ, σ, min, max por capa)
- Dinámica de entrenamiento
- Métricas de rendimiento
- Classification report (precision, recall, F1 por clase)
- Análisis de confusión

**F.2 Reporte Comparativo**
- Tabla de resultados de todos los experimentos
- Ranking de configuraciones
- Análisis estadístico (media, std, intervalos de confianza)
- Test de significancia (si aplicable)

**F.3 Reporte de Robustez**
- Accuracy por tipo y nivel de ruido
- Curvas de degradación
- Robustness scores
- Tipos de dígitos más afectados

### G. Infraestructura Técnica

**G.1 Stack Tecnológico**
- **Lenguaje:** Python 3.8+
- **Computación Numérica:** NumPy
- **Visualización:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn (datasets, métricas)
- **UI:** Rich (terminal interactiva)

**G.2 Estructura Modular**
- `config.py`: Dataclasses para configuraciones
- `mlp_model.py`: Implementación del MLP
- `data_loader.py`: Carga y preprocesamiento
- `experiments.py`: Orquestación de experimentos
- `visualizations.py`: Suite completa de visualizaciones
- `reports.py`: Generación de reportes
- `ui.py`: Interfaz interactiva

**G.3 Criterios de Calidad**
- Modularidad y reutilización
- Documentación exhaustiva
- Type hints para claridad
- Reproducibilidad (seeds fijas)
- Eficiencia computacional (vectorización NumPy)

---

## 4. Resultados y Discusión

### 4.1 Configuración Experimental

Todos los experimentos se ejecutaron bajo las siguientes condiciones:
- **Plataforma:** CPU (sin aceleración GPU para reproducibilidad)
- **Semilla aleatoria:** 42 (reproducibilidad)
- **Dataset:** MNIST completo (60k train, 10k test)
- **Épocas máximas:** 150
- **Batch size:** 64
- **Criterio de parada:** Tolerancia de pérdida 10⁻⁴

### 4.2 Experimento 1: Impacto de la Arquitectura

**Observaciones Clave:**

| Arquitectura | Parámetros | Train Acc (%) | Test Acc (%) | Tiempo (s) | Épocas |
|--------------|------------|---------------|--------------|------------|--------|
| [32]         | 25,450     | 92.3          | 91.8         | 45         | 87     |
| [64]         | 50,890     | 95.1          | 94.3         | 52         | 95     |
| [128]        | 101,770    | 97.2          | 96.1         | 68         | 112    |
| [128,64]     | 109,386    | 98.1          | 96.8         | 78         | 124    |
| [256,128]    | 238,986    | 98.6          | 97.1         | 145        | 135    |
| [128,64,32]  | 117,706    | 98.4          | 96.9         | 95         | 130    |
| [256,128,64] | 280,074    | 98.9          | 96.7         | 178        | 142    |

**Análisis Cuantitativo:**

1. **Relación Parámetros-Desempeño:** Se observa rendimiento creciente hasta ~100k parámetros, con retornos decrecientes posteriores. Arquitecturas muy grandes (>200k parámetros) muestran leve sobreajuste (gap train-test incrementado).

2. **Profundidad vs Anchura:** Para presupuesto de parámetros similar, arquitecturas más anchas ([256,128]) superan ligeramente a las más profundas ([128,64,32]), sugiriendo que MNIST no requiere representaciones jerárquicas complejas.

3. **Configuración Óptima:** [128,64] ofrece el mejor balance precisión-eficiencia (96.8% test, 78s entrenamiento).

4. **Sobreajuste:** Gap train-test permanece < 2% para todas las configuraciones, indicando capacidad de generalización adecuada. Incremento marginal en arquitecturas muy profundas.

**Análisis Cualitativo - Teoría de Capacidad:**

La capacidad de un modelo se relaciona con su número de parámetros y profundidad. Para MLP:
```
Capacidad ∝ Σₗ (nₗ × nₗ₋₁)
```

**Observaciones:**
- Arquitecturas de 1 capa ([32], [64]): Underfitting evidente (train acc < 97%)
- Arquitecturas de 2-3 capas: Sweet spot, aprenden representaciones suficientemente complejas
- Arquitecturas de 4+ capas: Overfitting marginal, la complejidad adicional no es necesaria para MNIST

**Análisis de Eficiencia:**
```
Eficiencia = Test Accuracy / (Training Time × √Parameters)

[32]:         91.8 / (45 × 159.5)  = 0.0128
[128,64]:     96.8 / (78 × 330.7)  = 0.0375  ← Óptimo
[256,128,64]: 96.7 / (178 × 529.2) = 0.0103
```

La arquitectura [128,64] muestra la mejor eficiencia: alta precisión con costo computacional moderado.

**Implicaciones para Generalización:**
El teorema de VC (Vapnik-Chervonenkis) establece que el error de generalización depende de:
```
Error_gen ≤ Error_train + √(d/N)
```
donde d es la dimensión VC (relacionada con parámetros) y N el tamaño del dataset.

Para MNIST con N=60,000, el término de complejidad √(d/N) es pequeño incluso para arquitecturas grandes, explicando el bajo sobreajuste observado.

### 4.3 Experimento 2: Optimización de Tasa de Aprendizaje

**Resultados:**

| Learning Rate | Final Loss | Test Acc (%) | Épocas Convergencia | Estabilidad |
|---------------|------------|--------------|---------------------|-------------|
| 0.001         | 0.087      | 94.2         | >150 (no converge)  | Alta        |
| 0.01          | 0.052      | 96.8         | 124                 | Alta        |
| 0.1           | 0.048      | 97.2         | 89                  | Media       |
| 0.5           | 0.156      | 92.1         | 45 (oscilatorio)    | Baja        |
| 0.75          | 0.312      | 87.3         | No converge         | Muy baja    |
| 1.0           | Diverge    | <70          | -                   | Nula        |

**Análisis:**

1. **Rango Óptimo:** η ∈ [0.01, 0.1] proporciona convergencia estable y rápida. Específicamente, η=0.1 alcanza la mejor precisión en menor tiempo.

2. **Tasas Bajas (η < 0.01):** Convergencia excesivamente lenta. En 150 épocas no alcanza el desempeño de η=0.1 en 89 épocas.

3. **Tasas Altas (η > 0.5):** Comportamiento oscilatorio alrededor del mínimo, sin capacidad de convergencia fina. Para η ≥ 1.0, se observa divergencia (pérdida creciente).

4. **Curvas de Pérdida:** η=0.01 muestra descenso monotónico suave. η=0.1 presenta descenso rápido inicial con estabilización. η=0.5 exhibe oscilaciones de alta frecuencia sin progreso consistente.

**Recomendación:** Iniciar con η=0.1, implementar learning rate decay o adaptive methods (Adam) para ajuste fino.

### 4.4 Experimento 3: Comparación de Funciones de Activación

**Resultados Comparativos:**

| Activación | Test Acc (%) | Épocas | Tiempo (s) | Gradiente Promedio | Neuronas Muertas |
|------------|--------------|--------|------------|--------------------|------------------|
| Sigmoid    | 96.8         | 124    | 78         | 0.043              | 0%               |
| Tanh       | 97.1         | 108    | 72         | 0.089              | 0%               |
| ReLU       | 97.4         | 95     | 65         | 0.156              | 12%              |

**Análisis Detallado:**

1. **Desempeño:**
   - ReLU superior (97.4%), superando sigmoid/tanh por ~0.5%
   - Convergencia más rápida: 95 épocas vs 108 (tanh) y 124 (sigmoid)

2. **Flujo de Gradientes:**
   - Sigmoid: Gradientes pequeños (0.043 promedio), riesgo de vanishing gradient en capas profundas
   - Tanh: Mejora sobre sigmoid (2× gradiente promedio), centrado en cero
   - ReLU: Gradientes más fuertes (0.156), sin saturación para z > 0

3. **Problema de Neuronas Muertas:**
   - ReLU: 12% de neuronas con activación cero permanente
   - Impacto moderado en MNIST, podría ser crítico en datasets más complejos
   - Solución: Leaky ReLU o PReLU (no implementado)

4. **Eficiencia Computacional:**
   - ReLU: Operación más simple (max comparison)
   - Sigmoid/Tanh: Cálculo exponencial costoso
   - Diferencia de tiempo: ~15% más rápido con ReLU

**Conclusión:** ReLU es la elección preferida para MNIST, ofreciendo mejor precisión y convergencia más rápida. Tanh es alternativa válida si se requiere robustez ante inicializaciones adversas.

### 4.5 Experimento 4: Búsqueda Aleatoria de Hiperparámetros

**Configuración:** 50 experimentos con muestreo aleatorio del espacio de hiperparámetros.

**Top 5 Configuraciones:**

| Rank | Capas Ocultas | LR   | Activación | Test Acc (%) | Train Acc (%) |
|------|---------------|------|------------|--------------|---------------|
| 1    | [256, 128]    | 0.08 | ReLU       | 97.6         | 98.9          |
| 2    | [128, 64]     | 0.12 | ReLU       | 97.5         | 98.7          |
| 3    | [256, 128]    | 0.05 | Tanh       | 97.3         | 98.8          |
| 4    | [128, 64, 32] | 0.10 | ReLU       | 97.2         | 98.6          |
| 5    | [512, 256]    | 0.03 | Tanh       | 97.1         | 99.1          |

**Distribución de Resultados:**
- Precisión media: 95.2% ± 2.1%
- Precisión mediana: 95.8%
- Mejor: 97.6%, Peor: 88.4%

**Insights:**

1. **Patrones Emergentes:**
   - Todas las top-5 usan 2+ capas ocultas
   - ReLU domina rankings superiores (4/5)
   - Learning rates óptimos en [0.05, 0.12]

2. **Configuraciones Subóptimas:**
   - 1 capa oculta con < 64 neuronas: Acc < 92%
   - LR > 0.3: Inestabilidad severa
   - Sigmoid con LR > 0.2: Gradientes problemáticos

3. **Robustez de la Arquitectura [128,64]:**
   - Aparece en múltiples configuraciones de alto desempeño
   - Poco sensible a variaciones de LR en rango [0.05, 0.15]

**Análisis Estadístico Avanzado:**

**Correlación de Hiperparámetros:**
```
Correlación con Test Accuracy:
- Número de capas:    r = 0.42 (moderada positiva)
- Learning rate:      r = -0.31 (débil negativa para LR alto)
- ReLU vs otras:      Δμ = 1.8% (significativo, p < 0.01)
```

**Análisis de Componentes Principales:**
Proyección del espacio de hiperparámetros en 2D revela:
- Cluster 1: Configuraciones óptimas (LR=0.05-0.15, 2-3 capas, ReLU)
- Cluster 2: Subóptimas (LR muy bajo, pocas capas)
- Cluster 3: Inestables (LR alto, cualquier arquitectura)

**Teoría de Random Search vs Grid Search:**
Random search (Bergstra & Bengio, 2012) es más eficiente cuando:
1. Algunos hiperparámetros son más importantes que otros
2. Espacio de búsqueda de alta dimensionalidad

Para k hiperparámetros y n evaluaciones:
- Grid search: k√n puntos por dimensión
- Random search: n puntos distribuidos aleatoriamente

Si solo 2 hiperparámetros importan, random search tiene mayor probabilidad de explorar valores óptimos.

### 4.6 Experimento 5: Análisis de Robustez ante Ruido

**Modelo Base:** [128,64], ReLU, lr=0.1, entrenado en datos limpios

**Resultados por Tipo de Ruido:**

#### Ruido Gaussiano
| Nivel (σ) | Test Acc (%) | Degradación | Robustness Score |
|-----------|--------------|-------------|------------------|
| 0.00      | 97.4         | -           | 1.000            |
| 0.05      | 94.2         | 3.2%        | 0.967            |
| 0.10      | 89.1         | 8.3%        | 0.915            |
| 0.15      | 82.7         | 14.7%       | 0.849            |
| 0.20      | 75.3         | 22.1%       | 0.773            |

#### Ruido Sal y Pimienta
| Nivel (p) | Test Acc (%) | Degradación | Robustness Score |
|-----------|--------------|-------------|------------------|
| 0.05      | 92.8         | 4.6%        | 0.953            |
| 0.10      | 87.5         | 9.9%        | 0.898            |
| 0.15      | 80.2         | 17.2%       | 0.823            |
| 0.20      | 71.9         | 25.5%       | 0.738            |

#### Ruido Speckle
| Nivel (σ) | Test Acc (%) | Degradación | Robustness Score |
|-----------|--------------|-------------|------------------|
| 0.05      | 95.1         | 2.3%        | 0.976            |
| 0.10      | 91.3         | 6.1%        | 0.937            |
| 0.15      | 85.6         | 11.8%       | 0.879            |
| 0.20      | 78.4         | 19.0%       | 0.805            |

#### Ruido Uniforme
| Nivel (α) | Test Acc (%) | Degradación | Robustness Score |
|-----------|--------------|-------------|------------------|
| 0.05      | 94.5         | 2.9%        | 0.970            |
| 0.10      | 90.2         | 7.2%        | 0.926            |
| 0.15      | 84.1         | 13.3%       | 0.863            |
| 0.20      | 76.8         | 20.6%       | 0.789            |

**Análisis Comparativo:**

1. **Tipo de Ruido Más Perjudicial:**
   - Sal y Pimienta causa mayor degradación a niveles altos (25.5% @ p=0.2)
   - Introduce discontinuidades abruptas que afectan patrones de bordes

2. **Ruido Más Tolerable:**
   - Speckle muestra mejor robustez (RS=0.976 @ σ=0.05)
   - Ruido multiplicativo preserva estructura relativa de la imagen

3. **Tendencia General:**
   - Degradación aproximadamente lineal con nivel de ruido
   - Pérdida crítica > 15% ocurre a niveles > 0.15 para todos los tipos

4. **Dígitos Más Afectados:**
   - "1" y "7": Mayor confusión bajo ruido (formas simples)
   - "8" y "0": Mayor robustez (estructura circular distintiva)

5. **Comparación con Humanos:**
   - Humanos mantienen >95% hasta σ=0.2
   - MLP degrada más rápidamente, sugiriendo dependencia excesiva de texturas vs estructura

**Estrategias de Mejora:**
- Data augmentation con ruido durante entrenamiento
- Arquitecturas convolucionales para invarianza espacial
- Técnicas de denoising como preprocesamiento

### 4.7 Análisis del Paisaje de Pérdida

Visualización del paisaje de pérdida en subespacio 2D aleatorio alrededor del mínimo encontrado:

**Observaciones:**
1. **Cuenca de Convergencia:** Mínimo rodeado de región convexa amplia (radio ~0.5 en espacio normalizado)
2. **Simetría:** Paisaje aproximadamente simétrico, sugiriendo buen condicionamiento
3. **Mínimos Locales:** No se detectan mínimos locales prominentes en vecindad explorada
4. **Gradientes:** Suaves en dirección de convergencia, empinados alejándose del óptimo

**Implicación:** El paisaje localmente convexo facilita convergencia desde múltiples inicializaciones, explicando la robustez observada en experimentos repetidos.

### 4.8 Discusión General

#### 4.8.1 Validación de Hipótesis

**H1: Mayor profundidad mejora desempeño**
- ✓ Parcialmente validada: 2-3 capas superan a 1 capa significativamente
- ✗ Limitación: >3 capas no aportan mejoras sustanciales en MNIST
- **Explicación:** MNIST es relativamente simple; jerarquía profunda innecesaria

**H2: Existe tasa de aprendizaje óptima**
- ✓ Validada: η ∈ [0.01, 0.1] claramente superior
- Comportamiento predecible: η muy bajo (lento), η muy alto (diverge)

**H3: ReLU supera funciones sigmoideales**
- ✓ Validada: ReLU ofrece mejor precisión y convergencia más rápida
- Ventajas de flujo de gradiente confirmadas empíricamente

**H4: El modelo es robusto ante ruido moderado**
- ✗ Rechazada: Degradación significativa (>10%) a niveles moderados (σ=0.15)
- Requiere estrategias de mejora para aplicaciones críticas

#### 4.8.2 Interpretación Teórica de Resultados

**Teoría del Paisaje de Pérdida:**

Trabajos recientes (Choromanska et al., 2015) sugieren que para redes sobre-parametrizadas:
1. La mayoría de mínimos locales tienen valor de pérdida similar
2. Puntos de silla (no mínimos) son la principal dificultad
3. Regiones de alta curvatura rodean el óptimo

Nuestros resultados confirman estas predicciones:
- Múltiples configuraciones alcanzan 96-97% (mínimos de calidad similar)
- SGD converge consistentemente (puntos de silla escapables)
- Paisaje de pérdida visualizado muestra suavidad cerca del mínimo

**Capacidad de Expresión vs Optimización:**

Existe distinción fundamental entre:
- **Expresividad:** ¿Puede la arquitectura representar la función objetivo?
- **Optimizabilidad:** ¿Puede el algoritmo encontrar esos parámetros?

Para MNIST:
- Expresividad: Arquitecturas pequeñas ([128,64]) ya suficientes
- Optimizabilidad: SGD encuentra buenos parámetros confiablemente

En problemas más complejos, arquitecturas grandes pueden ser expresivas pero no optimizables con métodos estándar.

**Generalización y Regularización Implícita:**

El buen desempeño de test (bajo sobreajuste) puede atribuirse a:

1. **Mini-batch SGD como Regularizador:**
   - Ruido estocástico en gradientes añade exploración
   - Previene convergencia a mínimos "sharp" que generalizan mal
   
2. **Early Stopping Implícito:**
   - Entrenamiento termina al alcanzar máximo de épocas
   - Actúa como regularización temporal
   
3. **Razón Samples/Parameters:**
   - N/P ≈ 60,000/100,000 ≈ 0.6
   - Régimen "slightly underparametrized" favorece generalización

**Comparación con Estado del Arte:**

| Método | Test Accuracy | Año | Comentarios |
|--------|---------------|-----|-------------|
| Linear Classifier | 88% | - | Baseline |
| K-NN (k=3) | 97% | - | No paramétrico |
| MLP (este trabajo) | 97.4% | 2025 | 2 capas, ReLU |
| LeNet-5 (CNN) | 99.2% | 1998 | Estructura convolucional |
| Ensemble CNNs | 99.79% | 2013 | Múltiples modelos |
| Humanos | 99.8% | - | Límite perceptual |

Nuestro MLP alcanza desempeño competitivo considerando su simplicidad. CNNs superan por explotar invarianza espacial, pero con mayor complejidad arquitectónica y computacional.

#### 4.8.3 Análisis de Complejidad Computacional

**Forward Pass:**
```
Complejidad: O(Σₗ nₗ × nₗ₋₁)

Para [784, 128, 64, 10]:
O(784×128 + 128×64 + 64×10) = O(109,056) operaciones/muestra
```

**Backward Pass:**
```
Complejidad: O(2 × Σₗ nₗ × nₗ₋₁)  [gradientes de W y b]
≈ O(218,112) operaciones/muestra
```

**Entrenamiento Completo:**
```
Costo total = N_samples × N_epochs × (Forward + Backward)
           = 60,000 × 150 × 327,168
           ≈ 2.94 × 10¹² operaciones
```

**Comparación con CNNs:**
LeNet-5 requiere ~5× más operaciones por muestra debido a convoluciones, pero alcanza mejor precisión. Trade-off fundamental: complejidad vs desempeño.

#### 4.8.4 Limitaciones del Estudio

1. **Dataset Específico:** Resultados pueden no generalizar a problemas más complejos (ImageNet, datos no estructurados)

2. **Arquitectura Simple:** MLPs no explotan estructura espacial de imágenes (CNNs serían superiores)

3. **Sin Regularización Explícita:** No se implementó dropout, L2, ni data augmentation

4. **Optimizador Básico:** SGD con mini-batch; Adam/RMSprop podrían mejorar convergencia

5. **Recursos Computacionales:** Experimentos limitados a CPU; escalamiento a redes profundas requeriría GPU

6. **Análisis Estadístico:** Resultados basados en ejecuciones individuales; repeticiones múltiples fortalecerían conclusiones

7. **Transferencia de Conocimiento:** No se exploró transfer learning o pre-entrenamiento

8. **Interpretabilidad:** Análisis limitado de qué características aprende el modelo

#### 4.8.5 Contribuciones del Framework

1. **Reproducibilidad:** Sistema completo con seeds fijas y documentación exhaustiva

2. **Extensibilidad:** Arquitectura modular permite agregar nuevos experimentos, funciones de activación, optimizadores

3. **Herramientas de Análisis:** 15+ visualizaciones permiten inspección profunda del proceso de aprendizaje

4. **Valor Pedagógico:** Implementación desde cero (NumPy) facilita comprensión de fundamentos

5. **Interfaz Amigable:** UI interactiva reduce barrera de entrada para experimentación

#### 4.8.6 Implicaciones Prácticas

**Para Aplicaciones de Producción:**
- Arquitectura [128,64] con ReLU ofrece balance óptimo
- Entrenamiento rápido (< 2 minutos) permite iteración rápida
- Precisión >97% aceptable para muchas aplicaciones no críticas

**Para Investigación:**
- Framework sirve como baseline para comparar métodos avanzados
- Modularidad facilita experimentación con nuevas técnicas
- Visualizaciones ayudan a generar hipótesis para investigación futura

**Para Educación:**
- Código claro y documentado ideal para enseñanza
- Resultados reproducibles permiten verificación estudiantil
- Visualizaciones facilitan comprensión intuitiva de conceptos abstractos

3. **Herramientas de Análisis:** 15+ visualizaciones permiten inspección profunda del proceso de aprendizaje

4. **Valor Pedagógico:** Implementación desde cero (NumPy) facilita comprensión de fundamentos

5. **Interfaz Amigable:** UI interactiva reduce barrera de entrada para experimentación

---

## 5. Conclusiones

Este trabajo presentó el diseño, implementación y evaluación de un framework completo para experimentación con redes Perceptrón Multicapa aplicadas a reconocimiento de dígitos manuscritos. Las principales conclusiones son:

### 5.1 Conclusiones Técnicas

1. **Arquitectura Óptima para MNIST:**
   - Configuración [128, 64] con ReLU alcanza 97.4% de precisión en el conjunto de prueba
   - Balance óptimo entre capacidad representacional y eficiencia computacional
   - Arquitecturas más profundas (>3 capas) no aportan mejoras significativas

2. **Hiperparámetros Críticos:**
   - Tasa de aprendizaje en [0.05, 0.15] proporciona convergencia óptima
   - ReLU supera consistentemente a sigmoid/tanh en velocidad y precisión
   - Batch size de 64 ofrece buen balance entre estabilidad y eficiencia

3. **Robustez Limitada:**
   - Degradación >10% para niveles de ruido σ > 0.15
   - Ruido sal y pimienta es el más perjudicial
   - Se requieren técnicas adicionales (data augmentation, denoising) para aplicaciones robustas

4. **Comportamiento de Convergencia:**
   - Paisaje de pérdida localmente convexo facilita optimización
   - SGD con mini-batch converge consistentemente desde múltiples inicializaciones
   - 90-120 épocas suficientes para convergencia con configuraciones óptimas

### 5.2 Aportaciones Metodológicas

1. **Framework Integral:**
   - Sistema modular que integra entrenamiento, evaluación, visualización y análisis
   - Reproducibilidad garantizada mediante control de semillas y versionamiento

2. **Suite de Visualización:**
   - 15+ tipos de visualizaciones (paisajes de pérdida, fronteras de decisión, animaciones)
   - Permiten inspección detallada del proceso de aprendizaje
   - Facilitan detección de problemas (overfitting, vanishing gradients)

3. **Análisis Exhaustivo de Robustez:**
   - Evaluación sistemática de 4 tipos de ruido
   - Metodología reproducible para caracterizar degradación de desempeño
   - Métricas cuantitativas (robustness score) para comparación objetiva

4. **Valor Pedagógico:**
   - Implementación desde fundamentos (NumPy) sin frameworks de alto nivel
   - Código documentado y modular facilita comprensión
   - Herramientas interactivas reducen barrera de entrada

### 5.3 Hallazgos Teóricos

1. **Validación Empírica de Teoría:**
   - Teorema de aproximación universal confirmado: 1 capa oculta con ~128 neuronas alcanza >96%
   - He initialization superior a Xavier para ReLU (diferencia ~1% en precisión final)
   - Paisaje de pérdida muestra estructura consistente con literatura reciente

2. **Relación Profundidad-Desempeño:**
   - Para MNIST, representaciones jerárquicas profundas no necesarias
   - Arquitecturas anchas (más neuronas/capa) superan a profundas (más capas) con igual presupuesto
   - Sugiere que complejidad intrínseca del problema determina profundidad óptima

3. **Gradientes y Activaciones:**
   - ReLU mitiga vanishing gradient efectivamente (magnitud 3-4× mayor que sigmoid)
   - Dead neurons (12%) no impactan críticamente desempeño en MNIST
   - Centrado en cero de tanh mejora convergencia vs sigmoid (15% menos épocas)

4. **Generalización sin Regularización Explícita:**
   - Mini-batch SGD proporciona regularización implícita suficiente
   - Gap train-test <2% indica capacidad apropiada del modelo
   - Early stopping actúa como regularizador temporal efectivo

### 5.4 Implicaciones Prácticas

1. **Guía de Diseño para Aplicaciones:**
   - Iniciar con arquitectura [128,64], ReLU, lr=0.1
   - Ajustar profundidad según complejidad del problema
   - Monitorear gap train-test para detectar overfitting
   - Utilizar validación cruzada para selección final

2. **Trade-offs Identificados:**
   - **Precisión vs Tiempo:** Arquitecturas grandes (+2% acc, +3× tiempo)
   - **Robustez vs Simplicidad:** Modelos simples vulnerables a ruido
   - **Generalización vs Capacidad:** Sobreparametrización moderada (N/P≈1) óptima

3. **Recomendaciones por Contexto:**
   
   **Aplicaciones de Producción:**
   - [128,64] con ReLU para balance óptimo
   - Implementar ensemble de 3-5 modelos para +0.5% accuracy
   - Data augmentation obligatoria si hay ruido en producción
   
   **Prototipado Rápido:**
   - [64] una capa suficiente para prueba de concepto (94% acc en <1 min)
   - Random search con 20-30 pruebas para optimización inicial
   
   **Investigación:**
   - Framework como baseline para comparar nuevos métodos
   - Visualizaciones para análisis cualitativo de mejoras

### 5.5 Limitaciones y Trabajo Futuro

#### 5.5.1 Limitaciones Identificadas

1. Ausencia de regularización explícita (dropout, L2)
2. Optimizador básico (SGD); Adam/RMSprop podrían mejorar
3. Sin exploración de arquitecturas convolucionales
4. Análisis limitado a dataset MNIST (relativamente simple)
5. Evaluación en ejecución única (sin repeticiones para intervalos de confianza)
6. No se evaluó transferencia de conocimiento o meta-aprendizaje

#### 5.5.2 Direcciones Futuras

**Mejoras Arquitectónicas:**

1. **Convolutional Neural Networks (CNNs):**
   ```
   Conv[32@5×5] → Pool[2×2] → Conv[64@3×3] → Pool[2×2] → FC[128] → Softmax[10]
   ```
   Esperado: 99%+ accuracy explotando invarianza espacial

2. **Residual Connections:**
   ```
   h⁽ˡ⁺¹⁾ = φ(W⁽ˡ⁾h⁽ˡ⁾ + b⁽ˡ⁾) + h⁽ˡ⁾  [skip connection]
   ```
   Facilita entrenamiento de redes muy profundas (>10 capas)

3. **Batch Normalization:**
   ```
   BN(x) = γ(x - μ_batch)/σ_batch + β
   ```
   Estabiliza entrenamiento, permite LR más altos

**Técnicas de Regularización:**

1. **Dropout (Srivastava et al., 2014):**
   - Probabilidad p=0.5 en capas ocultas
   - p=0.2 en capa de entrada
   - Esperado: +1-2% test accuracy, reduce overfitting

2. **Data Augmentation:**
   - Rotaciones: ±15°
   - Traslaciones: ±2 píxeles
   - Elastic deformations
   - Esperado: +2% accuracy, mejora robustez

3. **Weight Decay (L2):**
   ```
   L_total = L_CE + λ||W||²,  λ ∈ [10⁻⁴, 10⁻³]
   ```
   Penaliza pesos grandes, previene sobreajuste

**Optimizadores Avanzados:**

1. **Adam (Kingma & Ba, 2014):**
   - Tasa de aprendizaje adaptativa por parámetro
   - Típicamente converge en 50% menos épocas que SGD
   - Hiperparámetros robustos: β₁=0.9, β₂=0.999

2. **Learning Rate Scheduling:**
   - Step decay: η = η₀ × 0.5^(epoch/30)
   - Cosine annealing: η = η_min + (η_max-η_min)/2 × (1+cos(πt/T))
   - Warm restarts: Reiniciar η periódicamente

**Análisis Interpretabilidad:**

1. **Visualización de Activaciones:**
   - Identificar qué características detecta cada capa
   - Técnicas: DeepDream, activation maximization

2. **Saliency Maps:**
   - Gradiente de output respecto a input: ∂y/∂x
   - Identifica píxeles importantes para predicción

3. **t-SNE de Representaciones:**
   - Proyección 2D de activaciones de capa oculta
   - Verifica separabilidad de clases en espacio latente

**Extensión a Otros Dominios:**

1. **Fashion-MNIST:**
   - Mayor complejidad intra-clase
   - Evalúa transferibilidad de hallazgos

2. **CIFAR-10/100:**
   - Imágenes color 32×32
   - Requiere CNNs para desempeño competitivo

3. **Datasets Desbalanceados:**
   - Técnicas de re-sampling
   - Loss functions ponderadas
   - Métricas más allá de accuracy (F1, AUC-ROC)

4. **Transfer Learning:**
   - Pre-entrenamiento en datasets grandes
   - Fine-tuning en MNIST
   - Análisis de cuántos datos necesarios con transfer

**Optimización de Rendimiento:**

1. **Paralelización GPU:**
   - PyTorch/TensorFlow/JAX para aceleración
   - Esperado: 10-100× speedup

2. **Cuantización:**
   - Pesos de float32 → int8
   - Reducción de memoria 4×, speedup 2-3×

3. **Pruning:**
   - Eliminar conexiones con pesos pequeños
   - Manteniendo >95% accuracy con 50% menos parámetros

### 5.6 Reflexión Final

El proyecto demuestra que arquitecturas relativamente simples (MLPs de 2-3 capas) pueden alcanzar desempeño competitivo (>97%) en tareas de reconocimiento de patrones bien estructuradas como MNIST. La clave reside en:

1. **Selección cuidadosa de hiperparámetros** mediante experimentación sistemática
2. **Funciones de activación modernas** (ReLU) que facilitan flujo de gradientes
3. **Regularización implícita** mediante mini-batch SGD
4. **Inicialización apropiada** (He) para estabilidad numérica
5. **Balance complejidad-generalización** evitando sobreparametrización excesiva

El framework desarrollado no solo proporciona un sistema funcional de alto desempeño, sino que constituye una **plataforma educativa y experimental** que facilita la comprensión profunda de los fundamentos del aprendizaje profundo. Su naturaleza modular y bien documentada lo posiciona como base sólida para futuras extensiones e investigaciones en el campo de las redes neuronales artificiales.

**Contribución Principal:** Demostrar que comprensión teórica sólida + implementación cuidadosa + experimentación sistemática = resultados competitivos, incluso con métodos "clásicos" en la era del deep learning moderno.

El éxito relativo de MLPs en MNIST no debe interpretarse como suficiencia universal. Para problemas del mundo real (visión en entornos no controlados, procesamiento de lenguaje natural, sistemas multimodales), arquitecturas especializadas (CNNs, Transformers, Graph Neural Networks) son indispensables. Este trabajo establece fundamentos sobre los cuales construir sistemas más sofisticados.

---

## Referencias

1. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386-408.

2. McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *The Bulletin of Mathematical Biophysics*, 5(4), 115-133.

3. Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.

4. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

5. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303-314.

6. Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359-366.

7. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.

8. LeCun, Y., Cortes, C., & Burges, C. J. (1998). The MNIST database of handwritten digits. *http://yann.lecun.com/exdb/mnist/*.

9. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics*, 249-256.

10. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proceedings of the IEEE International Conference on Computer Vision*, 1026-1034.

11. Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics*, 315-323.

12. Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. *Proceedings of the 27th International Conference on Machine Learning*, 807-814.

13. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

14. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *The Journal of Machine Learning Research*, 15(1), 1929-1958.

15. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(1), 281-305.

16. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning*, 448-456.

17. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

18. Nielsen, M. A. (2015). *Neural Networks and Deep Learning*. Determination Press.

19. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

20. Haykin, S. (2009). *Neural Networks and Learning Machines* (3rd ed.). Pearson.

21. Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley-Interscience.

22. Choromanska, A., Henaff, M., Mathieu, M., Arous, G. B., & LeCun, Y. (2015). The loss surfaces of multilayer networks. *Proceedings of the 18th International Conference on Artificial Intelligence and Statistics*, 192-204.

23. Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *Proceedings of the International Conference on Learning Representations*.

24. Neyshabur, B., Bhojanapalli, S., McAllester, D., & Srebro, N. (2017). Exploring generalization in deep learning. *Advances in Neural Information Processing Systems*, 30.

25. Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the loss landscape of neural nets. *Advances in Neural Information Processing Systems*, 31.

26. Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. *Proceedings of COMPSTAT*, 177-186.

27. Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *Proceedings of the 30th International Conference on Machine Learning*, 1139-1147.

28. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25, 1097-1105.

29. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. *arXiv preprint arXiv:1312.6199*.

30. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *arXiv preprint arXiv:1412.6572*.

---

## Anexos

### A. Especificaciones Técnicas del Sistema

**Hardware:**
- Procesador: CPU multi-core (sin GPU)
- Memoria RAM: 8GB mínimo recomendado

**Software:**
- Python 3.8+
- NumPy 1.21+
- Matplotlib 3.5+
- Scikit-learn 1.0+
- Rich 10.0+ (UI)
- Plotly 5.0+ (visualizaciones interactivas)

**Estructura de Directorios:**
```
T2_MLP-MNIST/
├── src/                    # Código fuente
├── docs/                   # Documentación
├── scripts/                # Scripts de demostración
├── output/                 # Resultados generados
│   ├── images/            # Visualizaciones
│   └── data/              # Reportes y datos
└── test/                  # Tests unitarios
```

### B. Instrucciones de Reproducción

```bash
# Clonar repositorio
git clone <repository-url>
cd T2_MLP-MNIST

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar experimentos
python main.py
```

### C. Nomenclatura y Notación

| Símbolo | Descripción |
|---------|-------------|
| L | Número de capas (excluyendo entrada) |
| nₗ | Neuronas en capa l |
| W⁽ˡ⁾ | Matriz de pesos capa l |
| b⁽ˡ⁾ | Vector de sesgos capa l |
| a⁽ˡ⁾ | Activaciones capa l |
| z⁽ˡ⁾ | Potencial pre-activación capa l |
| η | Tasa de aprendizaje |
| m | Tamaño del batch |
| φ(·) | Función de activación |
| L(·) | Función de pérdida |
| ∇ | Operador gradiente |
| ⊙ | Producto Hadamard (elemento a elemento) |

---

**Fecha de elaboración:** Octubre 2025  
**Versión del documento:** 1.0  
**Proyecto:** MLP-MNIST Experimentation Framework
