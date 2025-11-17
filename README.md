# 📊 Visualización Interactiva de UMAP

Una aplicación web interactiva para visualizar y explorar la reducción de dimensionalidad usando **UMAP (Uniform Manifold Approximation and Projection)**. Permite cargar diferentes datasets, ajustar parámetros en tiempo real y comparar resultados.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Características

- ✅ **Múltiples datasets**: Iris, Wine, MNIST y archivos CSV personalizados
- ✅ **Visualización interactiva**: Gráficos 2D y 3D con Plotly
- ✅ **Parámetros ajustables**: Control total sobre `n_neighbors`, `min_dist`, `metric`, etc.
- ✅ **Comparación de implementaciones**: UMAP oficial vs implementación personalizada
- ✅ **Validación robusta**: Manejo completo de excepciones
- ✅ **Interfaz intuitiva**: Diseño moderno con Streamlit
- ✅ **Descarga de resultados**: Exporta embeddings como CSV

## 📋 Requisitos

### Versión de Python
- **Python 3.10, 3.11, 3.12 o 3.13**
- ⚠️ **NO compatible con Python 3.14** (debido a dependencia `numba`)

### Dependencias
Todas las dependencias están listadas en `requirements.txt`:
- `streamlit>=1.28.0` - Interfaz web
- `numpy>=1.24.0` - Operaciones numéricas
- `pandas>=2.0.0` - Manipulación de datos
- `scikit-learn>=1.3.0` - Datasets y preprocesamiento
- `umap-learn>=0.5.4` - Algoritmo UMAP oficial
- `matplotlib>=3.7.0` - Visualización
- `plotly>=5.17.0` - Gráficos interactivos
- `pytest>=7.4.0` - Testing (opcional)

## 🚀 Instalación

### 1. Clonar o descargar el proyecto

```bash
cd /ruta/a/tu/proyecto
```

### 2. Crear entorno virtual

```bash
# Con Python 3.12 (recomendado)
python3.12 -m venv venv

# O con la versión de Python que tengas (3.10-3.13)
python3 -m venv venv
```

### 3. Activar entorno virtual

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Tiempo estimado**: 2-5 minutos dependiendo de la conexión a internet.

## 💻 Uso

### Ejecutar la aplicación

```bash
# Asegúrate de estar en el directorio del proyecto
cd /Users/katalina/code/UMAP

# Activa el entorno virtual
source venv/bin/activate

# Ejecuta la aplicación
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

### Uso de la interfaz

1. **Seleccionar dataset**:
   - **Iris**: Dataset clásico de flores (150 muestras, 4 características, 3 clases)
   - **Wine**: Dataset de vinos (178 muestras, 13 características, 3 clases)
   - **MNIST (muestra)**: Muestra de dígitos escritos a mano (configurable, 784 características)
   - **Cargar archivo CSV**: Sube tu propio dataset

2. **Ajustar parámetros UMAP**:
   - **Dimensiones de salida**: 2D o 3D
   - **n_neighbors**: 2-200 (controla estructura local vs global)
   - **min_dist**: 0.0-1.0 (controla qué tan apretados están los clusters)
   - **Métrica**: euclidean, manhattan, cosine, etc.
   - **Random State**: Semilla para reproducibilidad

3. **Ejecutar UMAP**:
   - Haz clic en "🚀 Ejecutar UMAP"
   - Visualiza los resultados interactivos
   - Descarga los embeddings como CSV

4. **Comparar implementaciones** (opcional):
   - Selecciona "Comparar ambas" para ver UMAP oficial vs personalizado lado a lado

## 📁 Estructura del Proyecto

```
UMAP/
├── app.py                      # Aplicación principal Streamlit
├── src/                        # Código fuente
│   ├── __init__.py
│   ├── loader.py              # Carga y validación de datasets
│   ├── reducer.py             # Aplicación de UMAP oficial
│   ├── visualizer.py          # Visualizaciones interactivas
│   └── exceptions.py          # Excepciones personalizadas
├── requirements.txt            # Dependencias
├── README.md                  # Este archivo
└── venv/                      # Entorno virtual (no versionado)
```

## 📊 Datasets Disponibles

### Iris Dataset
- **Muestras**: 150
- **Características**: 4 (sepal length, sepal width, petal length, petal width)
- **Clases**: 3 (setosa, versicolor, virginica)
- **Fuente**: scikit-learn

### Wine Dataset
- **Muestras**: 178
- **Características**: 13 (componentes químicos del vino)
- **Clases**: 3 (tipos de vino)
- **Fuente**: scikit-learn

### MNIST Dataset (muestra)
- **Muestras**: Configurable (100-5000)
- **Características**: 784 (28x28 píxeles)
- **Clases**: 10 (dígitos 0-9)
- **Fuente**: OpenML (descarga automática)

### CSV Personalizado
- Carga tu propio archivo CSV
- Detección automática de columnas numéricas
- Soporte para columnas de etiquetas (opcional)
- Validación automática de datos

## 🎛️ Parámetros de UMAP

### `n_neighbors` (2-200, default: 15)
- **Efecto**: Controla cómo UMAP equilibra la estructura local vs global
- **Valores bajos** (2-10): Enfatiza la estructura local, clusters más compactos
- **Valores altos** (50-200): Enfatiza la estructura global, preserva más la forma general

### `min_dist` (0.0-1.0, default: 0.1)
- **Efecto**: Distancia mínima entre puntos en el espacio de baja dimensión
- **Valores bajos** (0.0-0.1): Puntos más apretados, clusters más densos
- **Valores altos** (0.5-1.0): Puntos más separados, clusters más dispersos

### `metric` (default: 'euclidean')
- **Opciones**: euclidean, manhattan, chebyshev, cosine, hamming, jaccard
- **Efecto**: Métrica utilizada para calcular distancias entre puntos
- **Recomendación**: 
  - `euclidean`: Para datos numéricos generales
  - `cosine`: Para datos de texto o cuando importa la dirección, no la magnitud
  - `manhattan`: Para datos con muchas dimensiones

### `random_state` (default: 42)
- **Efecto**: Semilla para reproducibilidad
- **Mismo valor**: Produce los mismos resultados cada vez
- **Diferente valor**: Produce resultados ligeramente diferentes

## 🔧 Uso Programático

También puedes usar los módulos directamente en Python:

```python
from src.loader import DatasetLoader
from src.reducer import UMAPReducer
from src.visualizer import UMAPVisualizer

# Cargar dataset
loader = DatasetLoader()
df, target, target_names, title = loader.load_iris()

# Preparar datos
X_scaled = loader.prepare_data(df, scale=True)

# Aplicar UMAP
reducer = UMAPReducer()
reducer.create_reducer(n_components=2, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(X_scaled)

# Visualizar
visualizer = UMAPVisualizer()
embedding_df = reducer.get_embedding_dataframe(target, target_names)
fig = visualizer.create_2d_plot(embedding_df, title)
fig.show()
```

## 🧪 Testing

Ejecutar tests con pytest:

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar tests
pytest
```

## 🛠️ Solución de Problemas

### Error: "Cannot install on Python version 3.14"
**Solución**: Usa Python 3.12 o 3.13
```bash
# Crear nuevo entorno con Python 3.12
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "ModuleNotFoundError: No module named 'src'"
**Solución**: Asegúrate de estar en el directorio raíz del proyecto
```bash
pwd  # Debe mostrar: /Users/katalina/code/UMAP
```

### La aplicación no se abre en el navegador
**Solución**: Abre manualmente `http://localhost:8501` en tu navegador

### Error al cargar CSV
**Solución**: 
- Verifica que el archivo tenga columnas numéricas
- Asegúrate de que el archivo no esté vacío
- Revisa que el formato CSV sea correcto

## 📚 Referencias

- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [UMAP Paper](https://arxiv.org/abs/1802.03426)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html)

