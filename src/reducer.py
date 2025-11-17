import numpy as np
import pandas as pd
import umap
from typing import Dict, Any
from .exceptions import ReducerError


class UMAPReducer:
    """Clase para aplicar UMAP a los datos."""
    
    def __init__(self):
        self.reducer = None
        self.embedding = None
        self.params = {}
    
    def create_reducer(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: int = 42,
        **kwargs
    ) -> umap.UMAP:

        if n_components not in [2, 3]:
            raise ReducerError("n_components debe ser 2 o 3")
        
        if n_neighbors < 2:
            raise ReducerError("n_neighbors debe ser al menos 2")
        
        if not (0.0 <= min_dist <= 1.0):
            raise ReducerError("min_dist debe estar entre 0.0 y 1.0")
        
        try:
            self.params = {
                'n_components': n_components,
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'metric': metric,
                'random_state': random_state,
                **kwargs
            }
            
            self.reducer = umap.UMAP(**self.params)
            return self.reducer
        except Exception as e:
            raise ReducerError(f"Error al crear el reductor UMAP: {str(e)}") from e
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.reducer is None:
            raise ReducerError("Debes crear un reductor primero usando create_reducer()")
        
        if not isinstance(X, np.ndarray):
            raise ReducerError("X debe ser un array numpy")
        
        if X.size == 0:
            raise ReducerError("El array de datos está vacío")
        
        if len(X.shape) != 2:
            raise ReducerError("X debe ser un array 2D (n_samples, n_features)")
        
        try:
            self.embedding = self.reducer.fit_transform(X)
            return self.embedding
        except Exception as e:
            raise ReducerError(f"Error al aplicar UMAP: {str(e)}") from e
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reducer is None:
            raise ValueError("El reductor no ha sido entrenado aún")
        
        return self.reducer.transform(X)
    
    def get_embedding_dataframe(
        self,
        target: np.ndarray = None,
        target_names: np.ndarray = None
    ) -> pd.DataFrame:

        if self.embedding is None:
            raise ReducerError("No hay embedding disponible. Ejecuta fit_transform() primero.")
        
        try:
            n_components = self.embedding.shape[1]
            columns = [f'UMAP {i+1}' for i in range(n_components)]
            
            embedding_df = pd.DataFrame(self.embedding, columns=columns)
            
            if target is not None:
                if len(target) != len(embedding_df):
                    raise ReducerError(
                        f"El tamaño de target ({len(target)}) no coincide con el número de muestras ({len(embedding_df)})"
                    )
                
                if target_names is not None:
                    embedding_df['Clase'] = [
                        target_names[i] if i < len(target_names) else str(i) 
                        for i in target
                    ]
                else:
                    embedding_df['Clase'] = target
            
            return embedding_df
        except Exception as e:
            if isinstance(e, ReducerError):
                raise
            raise ReducerError(f"Error al crear el DataFrame: {str(e)}") from e
    
    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()
    
    def get_reduction_stats(self, original_dim: int) -> Dict[str, Any]:
        if self.embedding is None:
            raise ReducerError("No hay embedding disponible. Ejecuta fit_transform() primero.")
        
        if original_dim <= 0:
            raise ReducerError("original_dim debe ser mayor que 0")
        
        try:
            reduced_dim = self.embedding.shape[1]
            reduction_pct = ((original_dim - reduced_dim) / original_dim) * 100
            
            return {
                'original_dim': original_dim,
                'reduced_dim': reduced_dim,
                'reduction_percentage': reduction_pct,
                'n_samples': self.embedding.shape[0]
            }
        except Exception as e:
            raise ReducerError(f"Error al calcular estadísticas: {str(e)}") from e

