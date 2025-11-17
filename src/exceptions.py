"""
Excepciones personalizadas para el proyecto UMAP.
"""


class UMAPError(Exception):
    """Excepción base para errores del proyecto UMAP."""
    pass


class DatasetError(UMAPError):
    """Excepción relacionada con la carga o validación de datasets."""
    pass


class ReducerError(UMAPError):
    """Excepción relacionada con la reducción de dimensionalidad."""
    pass


class VisualizationError(UMAPError):
    """Excepción relacionada con la visualización."""
    pass

