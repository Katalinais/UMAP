import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


class UMAPVisualizer:
    """Clase para crear visualizaciones de los embeddings de UMAP."""
    
    def __init__(self):
        self.fig = None
    
    def create_2d_plot(
        self,
        embedding_df: pd.DataFrame,
        dataset_title: str,
        color_column: Optional[str] = 'Clase',
        title_prefix: str = "UMAP"
    ) -> go.Figure:
        """
        Crea una visualización 2D del embedding.
        
        Args:
            embedding_df: DataFrame con el embedding (debe tener columnas 'UMAP 1' y 'UMAP 2')
            dataset_title: Título del dataset
            color_column: Nombre de la columna para colorear los puntos (None para no colorear)
            title_prefix: Prefijo para el título (por defecto "UMAP")
        
        Returns:
            Figura de Plotly
        """
        x_col = [col for col in embedding_df.columns if '1' in col and title_prefix.upper() in col.upper()][0]
        y_col = [col for col in embedding_df.columns if '2' in col and title_prefix.upper() in col.upper()][0]
        
        if color_column and color_column in embedding_df.columns:
            fig = px.scatter(
                embedding_df,
                x=x_col,
                y=y_col,
                color=color_column,
                title=f'Visualización {title_prefix} 2D - {dataset_title}',
                labels={x_col: f'{title_prefix} Component 1', 
                       y_col: f'{title_prefix} Component 2'},
                hover_data={color_column: True} if color_column else None
            )
        else:
            fig = px.scatter(
                embedding_df,
                x=x_col,
                y=y_col,
                title=f'Visualización {title_prefix} 2D - {dataset_title}',
                labels={x_col: f'{title_prefix} Component 1', 
                       y_col: f'{title_prefix} Component 2'}
            )
        
        fig.update_layout(height=600)
        self.fig = fig
        return fig
    
    def create_3d_plot(
        self,
        embedding_df: pd.DataFrame,
        dataset_title: str,
        color_column: Optional[str] = 'Clase',
        title_prefix: str = "UMAP"
    ) -> go.Figure:
        """
        Crea una visualización 3D del embedding.
        
        Args:
            embedding_df: DataFrame con el embedding (debe tener columnas 'UMAP 1', 'UMAP 2', 'UMAP 3')
            dataset_title: Título del dataset
            color_column: Nombre de la columna para colorear los puntos (None para no colorear)
            title_prefix: Prefijo para el título (por defecto "UMAP")
        
        Returns:
            Figura de Plotly
        """
        x_col = [col for col in embedding_df.columns if '1' in col and title_prefix.upper() in col.upper()][0]
        y_col = [col for col in embedding_df.columns if '2' in col and title_prefix.upper() in col.upper()][0]
        z_col = [col for col in embedding_df.columns if '3' in col and title_prefix.upper() in col.upper()][0]
        
        if color_column and color_column in embedding_df.columns:
            fig = px.scatter_3d(
                embedding_df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color_column,
                title=f'Visualización {title_prefix} 3D - {dataset_title}',
                labels={x_col: f'{title_prefix} Component 1', 
                       y_col: f'{title_prefix} Component 2',
                       z_col: f'{title_prefix} Component 3'}
            )
        else:
            fig = px.scatter_3d(
                embedding_df,
                x=x_col,
                y=y_col,
                z=z_col,
                title=f'Visualización {title_prefix} 3D - {dataset_title}',
                labels={x_col: f'{title_prefix} Component 1', 
                       y_col: f'{title_prefix} Component 2',
                       z_col: f'{title_prefix} Component 3'}
            )
        
        fig.update_layout(height=600)
        self.fig = fig
        return fig
    
    def create_plot(
        self,
        embedding_df: pd.DataFrame,
        dataset_title: str,
        n_components: int,
        color_column: Optional[str] = 'Clase',
        title_prefix: str = "UMAP"
    ) -> go.Figure:
        """
        Crea una visualización 2D o 3D según el número de componentes.
        
        Args:
            embedding_df: DataFrame con el embedding
            dataset_title: Título del dataset
            n_components: Número de componentes (2 o 3)
            color_column: Nombre de la columna para colorear los puntos
            title_prefix: Prefijo para el título
        
        Returns:
            Figura de Plotly
        """
        if n_components == 2:
            return self.create_2d_plot(embedding_df, dataset_title, color_column, title_prefix)
        elif n_components == 3:
            return self.create_3d_plot(embedding_df, dataset_title, color_column, title_prefix)
        else:
            raise ValueError("n_components debe ser 2 o 3")
    
    def create_class_distribution_chart(
        self,
        target: np.ndarray,
        target_names: Optional[np.ndarray] = None
    ) -> go.Figure:
        """
        Crea un gráfico de barras con la distribución de clases.
        
        Args:
            target: Array con las etiquetas
            target_names: Array con los nombres de las clases (opcional)
        
        Returns:
            Figura de Plotly
        """
        class_counts = pd.Series(target).value_counts().sort_index()
        
        if target_names is not None:
            class_counts.index = [target_names[i] if i < len(target_names) else str(i) 
                                 for i in class_counts.index]
        
        fig = px.bar(
            x=class_counts.index.astype(str),
            y=class_counts.values,
            title="Distribución de clases",
            labels={'x': 'Clase', 'y': 'Frecuencia'}
        )
        
        return fig

