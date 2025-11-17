    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris, load_wine, fetch_openml
    from sklearn.preprocessing import StandardScaler
    from typing import Tuple, Optional
    from .exceptions import DatasetError


    class DatasetLoader:
        """Clase para cargar y validar datasets."""
        
        def __init__(self):
            self.scaler = StandardScaler()
            self.df = None
            self.target = None
            self.target_names = None
            self.dataset_title = None
        
        ##Cargamos el dataset Iris
        def load_iris(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
            try:
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                target = data.target
                target_names = data.target_names
                return df, target, target_names, "Iris Dataset"
            except Exception as e:
                raise DatasetError(f"Error al cargar el dataset Iris: {str(e)}") from e
        
        ##Cargamos el dataset Wine
        def load_wine(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
            try:
                data = load_wine()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                target = data.target
                target_names = data.target_names
                return df, target, target_names, "Wine Dataset"
            except Exception as e:
                raise DatasetError(f"Error al cargar el dataset Wine: {str(e)}") from e
        
        ##Cargamos el dataset MNIST
        def load_mnist_sample(self, sample_size: int = 1000) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:

            if sample_size < 1:
                raise DatasetError("sample_size debe ser al menos 1")
            
            if sample_size > 70000:
                raise DatasetError("sample_size no puede ser mayor que 70000 (tamaño completo de MNIST)")
            
            try:
                mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
                
                if sample_size > len(mnist.data):
                    raise DatasetError(
                        f"sample_size ({sample_size}) es mayor que el tamaño del dataset ({len(mnist.data)})"
                    )
                
                # Tomar una muestra aleatoria
                indices = np.random.choice(len(mnist.data), sample_size, replace=False)
                X_sample = mnist.data[indices]
                y_sample = mnist.target[indices].astype(int)
                
                # Crear DataFrame con nombres de columnas
                feature_names = [f"pixel_{i}" for i in range(X_sample.shape[1])]
                df = pd.DataFrame(X_sample, columns=feature_names)
                target = y_sample
                target_names = [str(i) for i in range(10)]
                return df, target, target_names, f"MNIST Dataset (muestra de {sample_size})"
            except Exception as e:
                if isinstance(e, DatasetError):
                    raise
                raise DatasetError(f"Error al cargar el dataset MNIST: {str(e)}") from e


        ##Cargamos un archivo CSV
        def load_csv(self, file_path: str, label_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray], str]:

            if not file_path:
                raise DatasetError("file_path no puede estar vacío")
            
            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                raise DatasetError(f"Archivo no encontrado: {file_path}")
            except pd.errors.EmptyDataError:
                raise DatasetError(f"El archivo CSV está vacío: {file_path}")
            except Exception as e:
                raise DatasetError(f"Error al leer el archivo CSV: {str(e)}") from e
            
            if df.empty:
                raise DatasetError(f"El archivo CSV está vacío: {file_path}")
            
            try:
                # Separar columnas numéricas y no numéricas
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
                
                if not numeric_cols:
                    raise DatasetError("El archivo CSV no contiene columnas numéricas")
                
                # Si se especifica una columna de etiquetas
                if label_column:
                    if label_column not in df.columns:
                        raise DatasetError(f"La columna '{label_column}' no existe en el archivo")
                    
                    if label_column in non_numeric_cols:
                        target = df[label_column].astype('category').cat.codes
                        target_names = df[label_column].unique()
                    else:
                        target = df[label_column].values
                        target_names = np.unique(target)
                    df = df[numeric_cols]
                elif non_numeric_cols:
                    # Usar la última columna no numérica como etiqueta
                    label_col = non_numeric_cols[-1]
                    target = df[label_col].astype('category').cat.codes
                    target_names = df[label_col].unique()
                    df = df[numeric_cols]
                else:
                    target = None
                    target_names = None
                    df = df[numeric_cols]
                
                return df, target, target_names, file_path.split('/')[-1]
            except Exception as e:
                if isinstance(e, DatasetError):
                    raise
                raise DatasetError(f"Error al procesar el archivo CSV: {str(e)}") from e
        
        def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
            """
            Valida que el dataset sea adecuado para UMAP.
            
            Args:
                df: DataFrame a validar
            
            Returns:
                Tupla con (es_válido, mensaje_error)
            
            Raises:
                DatasetError: Si el DataFrame no es válido
            """
            if not isinstance(df, pd.DataFrame):
                raise DatasetError("El argumento debe ser un DataFrame de pandas")
            
            if df.empty:
                return False, "El dataset está vacío."
            
            if df.shape[0] < 2:
                return False, "El dataset debe tener al menos 2 muestras."
            
            if df.shape[1] < 2:
                return False, "El dataset debe tener al menos 2 características."
            
            # Verificar valores NaN
            try:
                if df.isnull().any().any():
                    nan_cols = df.columns[df.isnull().any()].tolist()
                    return False, f"El dataset contiene valores NaN en las columnas: {', '.join(nan_cols)}"
            except Exception as e:
                return False, f"Error al verificar valores NaN: {str(e)}"
            
            return True, None
        
        def prepare_data(self, df: pd.DataFrame, scale: bool = True) -> np.ndarray:

            if not isinstance(df, pd.DataFrame):
                raise DatasetError("El argumento debe ser un DataFrame de pandas")
            
            if df.empty:
                raise DatasetError("No se pueden preparar datos de un DataFrame vacío")
            
            try:
                if scale:
                    return self.scaler.fit_transform(df)
                else:
                    return df.values
            except Exception as e:
                raise DatasetError(f"Error al preparar los datos: {str(e)}") from e
        
        
        def load_dataset(self, dataset_name: str, **kwargs) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray], str]:

            if not dataset_name:
                raise DatasetError("dataset_name no puede estar vacío")
            
            try:
                if dataset_name == "Iris":
                    df, target, target_names, title = self.load_iris()
                elif dataset_name == "Wine":
                    df, target, target_names, title = self.load_wine()
                elif dataset_name == "MNIST (muestra)":
                    sample_size = kwargs.get('sample_size', 1000)
                    df, target, target_names, title = self.load_mnist_sample(sample_size)
                else:
                    # Asumimos que es una ruta a un archivo CSV
                    label_column = kwargs.get('label_column', None)
                    df, target, target_names, title = self.load_csv(dataset_name, label_column)
                
                # Validar el dataset
                is_valid, error_msg = self.validate_dataset(df)
                if not is_valid:
                    raise DatasetError(f"Dataset inválido: {error_msg}")
                
                self.df = df
                self.target = target
                self.target_names = target_names
                self.dataset_title = title
                
                return df, target, target_names, title
            except Exception as e:
                if isinstance(e, DatasetError):
                    raise
                raise DatasetError(f"Error al cargar el dataset '{dataset_name}': {str(e)}") from e
        
        def get_dataset_info(self) -> dict:
            if self.df is None:
                return {}
            
            info = {
                'title': self.dataset_title,
                'samples': len(self.df),
                'features': self.df.shape[1],
                'shape': self.df.shape
            }
            
            if self.target is not None:
                info['classes'] = len(np.unique(self.target))
                info['class_names'] = self.target_names.tolist() if self.target_names is not None else None
            
            return info

