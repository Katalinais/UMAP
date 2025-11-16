import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class DatasetLoader:
    """Clase para cargar y validar datasets."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.df = None
        self.target = None
        self.target_names = None
        self.dataset_title = None
    
    def load_iris(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
        """Carga el dataset Iris."""
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = data.target
        target_names = data.target_names
        return df, target, target_names, "Iris Dataset"
    
    def load_wine(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
        """Carga el dataset Wine."""
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = data.target
        target_names = data.target_names
        return df, target, target_names, "Wine Dataset"
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        if df.empty:
            return False, "El dataset está vacío."
        
        if df.shape[0] < 2:
            return False, "El dataset debe tener al menos 2 muestras."
        
        if df.shape[1] < 2:
            return False, "El dataset debe tener al menos 2 características."
        
        # Verificar valores NaN
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            return False, f"El dataset contiene valores NaN en las columnas: {', '.join(nan_cols)}"
        
        return True, None
    
    def prepare_data(self, df: pd.DataFrame, scale: bool = True) -> np.ndarray:

        if scale:
            return self.scaler.fit_transform(df)
        else:
            return df.values
