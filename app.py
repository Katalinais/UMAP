"""
Aplicación Streamlit para visualización interactiva de UMAP.
"""
import streamlit as st
import numpy as np
import pandas as pd
from src.loader import DatasetLoader
from src.reducer import UMAPReducer
from src.visualizer import UMAPVisualizer

# Configuración de la página
st.set_page_config(
    page_title="Visualización UMAP",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visualización Interactiva de UMAP")
st.markdown("""
Esta aplicación demuestra el funcionamiento de **UMAP (Uniform Manifold Approximation and Projection)** 
para la reducción de dimensionalidad. Puedes cargar diferentes datasets y ajustar los parámetros 
para ver cómo afectan la visualización.
""")

# Inicializar componentes
loader = DatasetLoader()
reducer = UMAPReducer()
visualizer = UMAPVisualizer()

# Sidebar para controles
st.sidebar.header("⚙️ Configuración")

# Selección de dataset
dataset_option = st.sidebar.selectbox(
    "Selecciona un dataset",
    ["Iris", "Wine"]
)

# Cargar dataset
df = None
target = None
target_names = None
dataset_title = None

try:
    if dataset_option == "Iris":
        df, target, target_names, dataset_title = loader.load_iris()
    elif dataset_option == "Wine":
        df, target, target_names, dataset_title = loader.load_wine()
    
    # Validar el dataset
    is_valid, error_msg = loader.validate_dataset(df)
    if not is_valid:
        st.error(f"Error en el dataset: {error_msg}")
        df = None
except Exception as e:
    st.error(f"Error al cargar el dataset: {str(e)}")
    st.exception(e)

# Mostrar información del dataset
if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Información del Dataset")
    st.sidebar.write(f"**Dataset:** {dataset_title}")
    st.sidebar.write(f"**Muestras:** {len(df)}")
    st.sidebar.write(f"**Características:** {df.shape[1]}")
    if target is not None:
        st.sidebar.write(f"**Clases:** {len(np.unique(target))}")
    
    # Parámetros de UMAP
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Parámetros de UMAP")
    
    n_components = st.sidebar.selectbox(
        "Dimensiones de salida",
        [2, 3],
        index=0,
        help="Número de dimensiones para la reducción (2D o 3D)"
    )
    
    n_neighbors = st.sidebar.slider(
        "n_neighbors",
        min_value=2,
        max_value=200,
        value=15,
        help="Controla cómo UMAP equilibra la estructura local vs global. Valores más bajos enfatizan la estructura local."
    )
    
    min_dist = st.sidebar.slider(
        "min_dist",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Distancia mínima entre puntos en el espacio de baja dimensión. Controla qué tan apretados están los clusters."
    )
    
    metric = st.sidebar.selectbox(
        "Métrica de distancia",
        ["euclidean", "manhattan", "chebyshev", "cosine", "hamming", "jaccard"],
        index=0,
        help="Métrica utilizada para calcular distancias entre puntos"
    )
    
    random_state = st.sidebar.number_input(
        "Random State",
        min_value=0,
        max_value=1000,
        value=42,
        help="Semilla para reproducibilidad"
    )
    
    # Botón para ejecutar UMAP
    run_umap = st.sidebar.button("🚀 Ejecutar UMAP", type="primary")
    
    # Mostrar datos originales
    st.subheader("📊 Datos Originales")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Forma del dataset:** {df.shape}")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        if target is not None:
            st.write("**Distribución de clases:**")
            fig_dist = visualizer.create_class_distribution_chart(target, target_names)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No hay etiquetas disponibles para este dataset.")
    
    # Ejecutar UMAP
    if run_umap or st.session_state.get('auto_run', False):
        with st.spinner("Ejecutando UMAP... Esto puede tardar unos segundos."):
            try:
                # Preparar datos
                X_scaled = loader.prepare_data(df, scale=True)
                
                # Crear y configurar reductor
                reducer.create_reducer(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    random_state=random_state
                )
                
                # Aplicar UMAP
                embedding = reducer.fit_transform(X_scaled)
                
                # Crear DataFrame con resultados
                embedding_df = reducer.get_embedding_dataframe(target, target_names)
                
                # Visualizar
                st.subheader("🎨 Visualización UMAP")
                fig = visualizer.create_plot(
                    embedding_df,
                    dataset_title,
                    n_components,
                    color_column='Clase' if target is not None else None,
                    title_prefix="UMAP"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar estadísticas
                stats = reducer.get_reduction_stats(df.shape[1])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dimensionalidad original", f"{stats['original_dim']}D")
                with col2:
                    st.metric("Dimensionalidad reducida", f"{stats['reduced_dim']}D")
                with col3:
                    st.metric("Reducción", f"{stats['reduction_percentage']:.1f}%")
                
                # Descargar resultados
                st.subheader("💾 Descargar Resultados")
                csv = embedding_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar embedding como CSV",
                    data=csv,
                    file_name=f"umap_embedding_{dataset_title.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Mostrar parámetros utilizados
                with st.expander("📝 Parámetros utilizados"):
                    params = reducer.get_params()
                    for key, value in params.items():
                        st.write(f"- **{key}:** {value}")
                
            except Exception as e:
                st.error(f"Error al ejecutar UMAP: {str(e)}")
                st.exception(e)
    else:
        st.info("👈 Ajusta los parámetros en la barra lateral y haz clic en 'Ejecutar UMAP' para ver la visualización.")
        st.session_state['auto_run'] = True

else:
    st.info("👈 Selecciona un dataset en la barra lateral para comenzar.")
