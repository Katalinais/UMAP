import numpy as np
import pandas as pd
import umap
from typing import Dict, Any, Tuple


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
            raise ValueError("n_components debe ser 2 o 3")
        
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
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        if self.reducer is None:
            raise ValueError("Debes crear un reductor primero usando create_reducer()")
        
        self.embedding = self.reducer.fit_transform(X)
        return self.embedding
    
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
            raise ValueError("No hay embedding disponible. Ejecuta fit_transform() primero.")
        
        n_components = self.embedding.shape[1]
        columns = [f'UMAP {i+1}' for i in range(n_components)]
        
        embedding_df = pd.DataFrame(self.embedding, columns=columns)
        
        if target is not None:
            if target_names is not None:
                embedding_df['Clase'] = [target_names[i] if i < len(target_names) else str(i) 
                                        for i in target]
            else:
                embedding_df['Clase'] = target
        
        return embedding_df
    
    def get_params(self) -> Dict[str, Any]:
        """Retorna los parámetros utilizados."""
        return self.params.copy()
    
    def get_reduction_stats(self, original_dim: int) -> Dict[str, Any]:

        if self.embedding is None:
            return {}
        
        reduced_dim = self.embedding.shape[1]
        reduction_pct = ((original_dim - reduced_dim) / original_dim) * 100
        
        return {
            'original_dim': original_dim,
            'reduced_dim': reduced_dim,
            'reduction_percentage': reduction_pct,
            'n_samples': self.embedding.shape[0]
        }

