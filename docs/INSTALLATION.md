# 🚀 Instrucciones de Instalación y Ejecución

## 📋 Requisitos Previos

- **Python 3.8 o superior**
- **pip** (gestor de paquetes de Python)
- **~500 MB** de espacio en disco
- **Conexión a Internet** (para descargar MNIST la primera vez)

## 🔧 Instalación Paso a Paso

### Paso 1: Verificar Python

Abre una terminal PowerShell y verifica tu versión de Python:

```powershell
python --version
```

Deberías ver algo como: `Python 3.8.x` o superior.

### Paso 2: Activar Entorno Virtual (Recomendado)

Si ya tienes un entorno virtual creado:

```powershell
# Windows PowerShell
.\venv\Scripts\activate

# O si usas CMD
.\venv\Scripts\activate.bat
```

Si no tienes entorno virtual, créalo:

```powershell
# Crear entorno virtual
python -m venv venv

# Activarlo
.\venv\Scripts\activate
```

Verás `(venv)` al inicio de tu terminal cuando esté activado.

### Paso 3: Instalar Dependencias

Instala todas las librerías necesarias:

```powershell
pip install -r requirements.txt
```

Este comando instalará:
- numpy (cálculos numéricos)
- scikit-learn (utilidades ML)
- matplotlib (gráficos)
- seaborn (visualizaciones estadísticas)
- plotly (dashboards interactivos)
- rich (interfaz terminal)
- pandas (manejo de datos)
- scipy (cálculos científicos)
- pillow (procesamiento de imágenes)
- imageio (animaciones)

**Tiempo estimado**: 2-5 minutos

### Paso 4: Verificar Instalación

Ejecuta el script de verificación:

```powershell
python verify_setup.py
```

Este script verifica:
- ✓ Versión de Python
- ✓ Todas las dependencias instaladas
- ✓ Estructura del proyecto
- ✓ Imports de módulos
- ✓ Funcionalidad básica

Si todo está ✓ (verde), ¡estás listo! Si hay ✗ (rojo), revisa los errores mostrados.

## 🎯 Primera Ejecución

### Opción 1: Demo Rápida (RECOMENDADO para empezar)

```powershell
python scripts/quick_start.py
```

**Lo que hace:**
- Carga 2000 muestras de MNIST
- Entrena un MLP básico [128, 64]
- Genera 7+ visualizaciones
- Crea un reporte matemático

**Tiempo**: ~2-3 minutos

**Salidas en**: `output/images/` y `output/data/`

### Opción 2: Aplicación Interactiva

```powershell
python main.py
```

**Lo que muestra:**
- Banner ASCII art grande
- Menú interactivo con 9 opciones
- Interfaz colorida y profesional

**Opciones del menú:**

1. **Quick Experiment** (2 min)
   - Experimento rápido con configuración por defecto
   
2. **Layer Configuration Analysis** (5-10 min)
   - Prueba 8 arquitecturas diferentes
   - Compara 1-4 capas ocultas
   
3. **Learning Rate Exploration** (5-10 min)
   - Prueba 6 tasas de aprendizaje
   - Desde 0.001 hasta 1.0
   
4. **Activation Function Comparison** (5 min)
   - Compara sigmoid vs tanh
   
5. **Comprehensive Grid Search** (15-30 min)
   - Búsqueda exhaustiva de hiperparámetros
   - Configurable número de configuraciones
   
6. **Noise Robustness Testing** (3-5 min)
   - Prueba con 4 tipos de ruido
   - Evalúa robustez del modelo
   
7. **Configure Experiment Settings**
   - Personaliza dataset, MLP, experimentos
   
8. **Load & Analyze Previous Results**
   - Ver resultados guardados
   
9. **Exit**

### Opción 3: Demostración de Ruido

```powershell
python scripts/noise_demo.py
```

**Lo que hace:**
- Muestra 4 tipos de ruido
- Genera comparaciones visuales
- Clean vs Gaussian, Salt&Pepper, Speckle, Uniform

**Tiempo**: ~1-2 minutos

### Opción 4: Experimentos Avanzados

```powershell
python scripts/advanced_experiment.py
```

**Lo que hace:**
- Experimento 1: Capas (8 configs)
- Experimento 2: Learning rates (5 valores)
- Experimento 3: Activaciones (2 funciones)
- Genera todas las visualizaciones
- Crea dashboard interactivo
- Genera reportes comparativos

**Tiempo**: ~15-20 minutos

### Opción 5: Batch (Todo Incluido)

```powershell
python scripts/batch_experiment.py
```

**Lo que hace:**
- Ejecuta TODOS los experimentos
- Genera TODAS las visualizaciones
- Crea TODOS los reportes
- Ideal para reporte final

**Tiempo**: ~30 minutos

## 📊 Ver Resultados

### Visualizaciones

```powershell
# Abrir carpeta de imágenes
explorer output\images

# O en PowerShell
cd output\images
ls
```

Encontrarás:
- `*.png` - Imágenes estáticas (gráficos)
- `*.gif` - Animaciones de entrenamiento
- `*.html` - Dashboards interactivos (ábrelos en navegador)

### Reportes

```powershell
# Abrir carpeta de datos
explorer output\data

# Ver reporte
notepad output\data\mathematical_report.txt
```

Encontrarás:
- `*.txt` - Reportes matemáticos
- `*.json` - Resultados serializados
- `*.pkl` - Cache de datos

### Dashboard Interactivo

```powershell
# Abrir en navegador por defecto
start output\images\experiment_dashboard.html
```

## 🔍 Explorar el Código

### Estructura Principal

```
src/
├── config.py          # 📝 Configuraciones
├── data_loader.py     # 📥 Carga de datos y ruido
├── mlp_model.py       # 🧠 Implementación del MLP
├── experiments.py     # 🔬 Sistema de experimentación
├── visualizations.py  # 📊 Suite de visualizaciones
├── reports.py         # 📈 Reportes matemáticos
└── ui.py             # 🎨 Interfaz interactiva
```

### Abrir en Editor

```powershell
# VS Code
code .

# O cualquier editor de texto
notepad src\mlp_model.py
```

## ⚙️ Configuración Personalizada

### Cambiar Tamaño del Dataset

Edita `main.py` o crea tu propio script:

```python
from src.config import DatasetConfig

# Más muestras = mejor precisión pero más lento
config = DatasetConfig(
    n_samples=10000,  # Por defecto: 5000
    test_size=0.2     # 20% para prueba
)
```

### Cambiar Arquitectura del MLP

```python
from src.config import MLPConfig

config = MLPConfig(
    hidden_layers=[256, 128, 64],  # Red más profunda
    learning_rate=0.01,            # LR balanceado
    activation="tanh",             # tanh > sigmoid
    max_epochs=100,                # Más épocas
    batch_size=64                  # Tamaño de lote
)
```

## 🐛 Solución de Problemas

### Error: `ModuleNotFoundError: No module named 'XXX'`

**Solución:**
```powershell
pip install XXX
# O reinstala todo:
pip install -r requirements.txt
```

### Error: `MNIST download fails`

**Solución:**
- El sistema reintentará automáticamente
- Verifica tu conexión a internet
- Si persiste, el dataset se cacheará tras el primer éxito

### Advertencia: `Slow training`

**Soluciones:**
1. Reducir n_samples:
   ```python
   DatasetConfig(n_samples=1000)
   ```

2. Arquitectura más simple:
   ```python
   MLPConfig(hidden_layers=[64, 32])
   ```

3. Menos épocas:
   ```python
   MLPConfig(max_epochs=30)
   ```

### Error: `Out of memory`

**Soluciones:**
1. Reducir batch_size:
   ```python
   MLPConfig(batch_size=32)
   ```

2. Reducir n_samples:
   ```python
   DatasetConfig(n_samples=2000)
   ```

### El script no responde

**Solución:**
- Presiona `Ctrl+C` para interrumpir
- El sistema preguntará si quieres salir
- Los resultados parciales se guardan automáticamente

## 📝 Para tu Reporte Académico

### 1. Ejecuta Batch Experiment

```powershell
python scripts/batch_experiment.py
```

Esto genera TODO lo necesario.

### 2. Recopila Resultados

```powershell
# Copia carpeta output a tu reporte
xcopy output\images docs\IMRAD\images /E /I
xcopy output\data docs\IMRAD\data /E /I
```

### 3. Usa las Visualizaciones

En tu reporte incluye:
- `training_history.png` - Para metodología
- `confusion_matrix.png` - Para resultados
- `comparison_report.txt` - Para análisis
- `experiment_dashboard.html` - Para presentación

### 4. Cita el Código

```
Ver código fuente en:
- src/mlp_model.py (líneas XX-XX) para backpropagation
- src/experiments.py (líneas XX-XX) para experimentación
```

## 🎓 Recursos de Aprendizaje

### Documentación

```powershell
# Abrir README
notepad README.md

# Guía rápida
notepad docs\QUICKSTART.md

# Guía de experimentos
notepad docs\EXPERIMENTS.md

# Resumen del proyecto
notepad docs\PROJECT_SUMMARY.md
```

### Código de Ejemplo

Ver:
- `scripts/quick_start.py` - Ejemplo simple
- `scripts/advanced_experiment.py` - Ejemplo completo
- `main.py` - Aplicación completa

## 🚀 Flujo de Trabajo Recomendado

### Día 1: Exploración (30 min)
```powershell
python verify_setup.py         # 1 min
python scripts/quick_start.py  # 2 min
python main.py                 # Explora menú
```

### Día 2: Experimentación (1-2 horas)
```powershell
python main.py
# Opción 2: Layer configs
# Opción 3: Learning rates
# Opción 4: Activations
```

### Día 3: Análisis de Ruido (30 min)
```powershell
python scripts/noise_demo.py
python main.py  # Opción 6
```

### Día 4: Análisis Completo (1 hora)
```powershell
python scripts/batch_experiment.py
# Revisar output/data/comparison_report.txt
# Abrir output/images/experiment_dashboard.html
```

### Día 5: Reporte (2-3 horas)
- Escribir IMRAD usando resultados
- Incluir visualizaciones
- Citar reportes matemáticos
- Justificar configuración óptima

## 📞 Ayuda Adicional

### Comandos Útiles

```powershell
# Ver archivos generados
ls output\images
ls output\data

# Limpiar outputs (cuidado!)
rm output\images\*.png
rm output\data\*.json

# Ver logs (si hay errores)
python main.py 2> errors.log

# Ejecutar tests
pytest test/test_mlp.py -v
```

### Verificar Estado

```powershell
python verify_setup.py
```

### Actualizar Dependencias

```powershell
pip install --upgrade -r requirements.txt
```

## ✅ Checklist Antes de Empezar

- [ ] Python 3.8+ instalado
- [ ] Entorno virtual activado (`venv`)
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Verificación pasada (`python verify_setup.py`)
- [ ] Quick start ejecutado exitosamente

## 🎉 ¡Listo para Empezar!

Una vez completada la instalación:

```powershell
# Inicia con la demo rápida
python scripts/quick_start.py

# Luego explora la aplicación interactiva
python main.py
```

**¡Buena suerte con tu proyecto! 🚀**

---

**¿Problemas?** Revisa:
1. `README.md` - Documentación completa
2. `docs/QUICKSTART.md` - Guía de uso
3. `verify_setup.py` - Diagnóstico de problemas
