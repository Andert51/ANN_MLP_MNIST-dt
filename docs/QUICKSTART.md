# 🚀 Guía Rápida de Uso

## Instalación Rápida

```bash
# 1. Activar entorno virtual (si existe)
.\venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Formas de Uso

### 📱 Modo 1: Aplicación Interactiva (RECOMENDADO)

La forma más fácil y completa:

```bash
python main.py
```

**Características:**
- ✨ Interfaz hermosa con Rich
- 📊 Menú interactivo
- ⚙️ Configuración guiada
- 📈 Resultados en tiempo real
- 🎨 Selección de visualizaciones

**Opciones del menú:**
1. **Quick Experiment** - Experimento rápido (2 min)
2. **Layer Configuration** - Análisis de capas (5-10 min)
3. **Learning Rate** - Exploración de LR (5-10 min)
4. **Activation Functions** - Comparación (5 min)
5. **Comprehensive Search** - Búsqueda exhaustiva (15-30 min)
6. **Noise Robustness** - Pruebas con ruido (3-5 min)
7. **Configure Settings** - Personalizar configuración
8. **Load Results** - Ver resultados previos

### 🎯 Modo 2: Script Rápido

Para una demostración simple:

```bash
python scripts/quick_start.py
```

**Genera:**
- 7+ visualizaciones
- Reporte matemático
- ~2 minutos de ejecución

### 🔬 Modo 3: Experimento Avanzado

Para análisis completo:

```bash
python scripts/advanced_experiment.py
```

**Genera:**
- 3 tipos de experimentos
- 15+ visualizaciones
- Reportes comparativos
- Dashboard interactivo
- ~15-20 minutos

### 📦 Modo 4: Batch (Todos los Experimentos)

Para ejecutar todo de una vez:

```bash
python scripts/batch_experiment.py
```

**Incluye:**
- Todos los experimentos
- Todas las visualizaciones
- Todos los reportes
- ~20-30 minutos

### 🔊 Modo 5: Demo de Ruido

Para ver comparación de tipos de ruido:

```bash
python scripts/noise_demo.py
```

## Ejemplos de Código

### Entrenar un Modelo Simple

```python
from src.config import MLPConfig
from src.mlp_model import MLPClassifier
from src.data_loader import MNISTLoader

# Cargar datos
loader = MNISTLoader()
X_train, X_test, y_train, y_test = loader.load_data()

# Configurar modelo
config = MLPConfig(
    hidden_layers=[128, 64],
    learning_rate=0.01,
    max_epochs=50
)

# Entrenar
model = MLPClassifier(config)
model.fit(X_train, y_train)

# Evaluar
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Ejecutar Experimentos

```python
from src.experiments import ExperimentRunner
from src.config import ExperimentConfig

config = ExperimentConfig()
runner = ExperimentRunner(config)

# Probar diferentes tasas de aprendizaje
results = runner.run_learning_rate_experiment(
    X_train, y_train, X_test, y_test
)

# Encontrar el mejor
best = runner.find_best_configuration(results)
print(f"Best LR: {best['config'].learning_rate}")
```

### Generar Visualizaciones

```python
from src.visualizations import MLPVisualizer
from src.config import VisualizationConfig

viz = MLPVisualizer(VisualizationConfig())

# Training history
viz.plot_training_history(
    model.history, 
    save_name="my_training.png"
)

# Confusion matrix
viz.plot_confusion_matrix(
    y_test, 
    model.predict(X_test),
    save_name="my_confusion.png"
)
```

### Agregar Ruido

```python
from src.data_loader import NoiseGenerator
from src.config import NoiseConfig

# Configurar ruido
noise_config = NoiseConfig(
    noise_type="gaussian",
    noise_level=0.2
)

# Aplicar ruido
X_noisy = NoiseGenerator.apply_noise(X_test, noise_config)

# Evaluar robustez
clean_acc = model.score(X_test, y_test)
noisy_acc = model.score(X_noisy, y_test)

print(f"Clean: {clean_acc:.4f}")
print(f"Noisy: {noisy_acc:.4f}")
print(f"Robustness: {noisy_acc/clean_acc:.4f}")
```

## Salidas Generadas

### Visualizaciones (output/images/)

```
01_dataset_samples.png          - Muestras del dataset
02_training_history.png         - Curvas de entrenamiento
03_confusion_matrix.png         - Matriz de confusión
04_predictions.png              - Predicciones con confianza
05_probability_heatmap.png      - Mapa de probabilidades
06_weight_distributions.png     - Distribución de pesos
07_decision_boundary.png        - Límites de decisión
08_loss_landscape.png           - Paisaje de pérdida
training_animation.gif          - Animación de entrenamiento
experiment_dashboard.html       - Dashboard interactivo
```

### Reportes (output/data/)

```
mathematical_report.txt         - Análisis matemático completo
comparison_report.txt           - Comparación de configuraciones
*_results.json                  - Resultados serializados
mnist_cache.pkl                - Cache de datos MNIST
```

## Configuración Personalizada

### Cambiar Tamaño del Dataset

```python
from src.config import DatasetConfig

config = DatasetConfig(
    n_samples=10000,  # Más datos = mejor modelo pero más lento
    test_size=0.2     # 20% para prueba
)
```

### Personalizar Arquitectura

```python
from src.config import MLPConfig

config = MLPConfig(
    hidden_layers=[512, 256, 128, 64],  # Red profunda
    learning_rate=0.001,                # LR bajo para estabilidad
    activation="tanh",                  # Tanh mejor que sigmoid
    max_epochs=200,                     # Más épocas si es necesario
    batch_size=128                      # Batch más grande = más rápido
)
```

### Configurar Experimentos

```python
from src.config import ExperimentConfig

config = ExperimentConfig(
    hidden_layer_configs=[
        [64], [128], [256],
        [128, 64], [256, 128]
    ],
    learning_rates=[0.001, 0.01, 0.1],
    activations=["sigmoid", "tanh"],
    max_epochs=100
)
```

## Tips de Performance

### 💨 Para Ejecución Rápida

```python
DatasetConfig(n_samples=1000)    # Menos muestras
MLPConfig(
    hidden_layers=[64, 32],      # Red más simple
    max_epochs=30,               # Menos épocas
    batch_size=64                # Batch medio
)
```

### 🎯 Para Mejor Precisión

```python
DatasetConfig(n_samples=10000)   # Más muestras
MLPConfig(
    hidden_layers=[256, 128, 64], # Red más profunda
    learning_rate=0.01,           # LR balanceado
    max_epochs=150,               # Más épocas
    batch_size=64                 # Batch medio
)
```

### ⚡ Para Experimentación Intensiva

```python
ExperimentConfig(
    hidden_layer_configs=[
        # 10+ configuraciones
    ],
    learning_rates=[
        # 6+ valores
    ],
    max_epochs=100,
    n_samples=5000
)
```

## Solución de Problemas

### Error: No module named 'src'

```bash
# Ejecutar desde la raíz del proyecto
cd T2_MLP-MNIST
python main.py
```

### Error: MNIST download fails

```bash
# El sistema intentará automáticamente varias veces
# Si persiste, verificar conexión a internet
```

### Advertencia: Slow training

```python
# Reducir n_samples
config = DatasetConfig(n_samples=1000)

# O usar arquitectura más simple
config = MLPConfig(hidden_layers=[64, 32])
```

### Error: Out of memory

```python
# Reducir batch_size
config = MLPConfig(batch_size=32)

# O reducir n_samples
config = DatasetConfig(n_samples=2000)
```

## Flujo de Trabajo Recomendado

### 1️⃣ Exploración Inicial (5 min)

```bash
python scripts/quick_start.py
```

- Ver qué hace el sistema
- Entender las visualizaciones
- Revisar reportes generados

### 2️⃣ Experimentación (15-30 min)

```bash
python main.py
```

- Opción 2: Layer configs
- Opción 3: Learning rates
- Opción 4: Activations

### 3️⃣ Análisis de Ruido (5 min)

```bash
python scripts/noise_demo.py
python main.py  # Opción 6
```

### 4️⃣ Análisis Completo (20-30 min)

```bash
python scripts/batch_experiment.py
```

- Genera TODO
- Perfecto para reporte final

### 5️⃣ Documentación

- Revisar `output/data/comparison_report.txt`
- Abrir `output/images/experiment_dashboard.html`
- Revisar visualizaciones en `output/images/`

## Estructura de Reporte Final

Para tu proyecto académico, incluye:

1. **Introducción**
   - Objetivo del proyecto
   - Dataset MNIST
   - Arquitectura MLP

2. **Metodología**
   - Configuraciones probadas
   - Hiperparámetros explorados
   - Métricas evaluadas

3. **Resultados**
   - Tablas de `comparison_report.txt`
   - Visualizaciones de `output/images/`
   - Dashboard HTML

4. **Análisis**
   - Mejor configuración
   - Trade-offs observados
   - Efectos de hiperparámetros

5. **Conclusiones**
   - Configuración óptima encontrada
   - Lecciones aprendidas
   - Recomendaciones

## Recursos Adicionales

- `README.md` - Documentación completa
- `docs/EXPERIMENTS.md` - Guía de experimentos
- `test/test_mlp.py` - Tests unitarios

## Soporte

Si encuentras problemas:

1. Verifica que todas las dependencias estén instaladas
2. Asegúrate de estar en el directorio correcto
3. Revisa los mensajes de error (son descriptivos)
4. Consulta el README.md para más detalles

---

**¡Éxito con tu proyecto! 🎓**
