"""
Script para probar loader.py y reducer.py sin Streamlit.
"""
import numpy as np
from src.loader import DatasetLoader
from src.reducer import UMAPReducer

def test_loader():
    """Prueba el DatasetLoader."""
    print("=" * 50)
    print("TESTING DatasetLoader")
    print("=" * 50)
    
    loader = DatasetLoader()
    
    # Test 1: Cargar Iris
    print("\n1. Cargando dataset Iris...")
    df, target, target_names, title = loader.load_iris()
    print(f"   ✓ Dataset cargado: {title}")
    print(f"   ✓ Forma: {df.shape}")
    print(f"   ✓ Clases: {target_names}")
    
    # Test 2: Validar dataset
    print("\n2. Validando dataset...")
    is_valid, error_msg = loader.validate_dataset(df)
    if is_valid:
        print(f"   ✓ Dataset válido")
    else:
        print(f"   ✗ Error: {error_msg}")
    
    # Test 3: Preparar datos
    print("\n3. Preparando datos...")
    X_scaled = loader.prepare_data(df, scale=True)
    print(f"   ✓ Datos preparados: {X_scaled.shape}")
    print(f"   ✓ Media (debe estar cerca de 0): {X_scaled.mean():.6f}")
    print(f"   ✓ Desviación estándar (debe estar cerca de 1): {X_scaled.std():.6f}")
    
    # Test 4: Cargar Wine
    print("\n4. Cargando dataset Wine...")
    df_wine, target_wine, target_names_wine, title_wine = loader.load_wine()
    print(f"   ✓ Dataset cargado: {title_wine}")
    print(f"   ✓ Forma: {df_wine.shape}")
    print(f"   ✓ Clases: {target_names_wine}")
    
    return df, target, target_names, X_scaled

def test_reducer(X_scaled, target, target_names):
    """Prueba el UMAPReducer."""
    print("\n" + "=" * 50)
    print("TESTING UMAPReducer")
    print("=" * 50)
    
    reducer = UMAPReducer()
    
    # Test 1: Crear reductor 2D
    print("\n1. Creando reductor UMAP 2D...")
    reducer.create_reducer(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    print("   ✓ Reductor creado")
    print(f"   ✓ Parámetros: {reducer.get_params()}")
    
    # Test 2: Aplicar UMAP
    print("\n2. Aplicando UMAP...")
    embedding = reducer.fit_transform(X_scaled)
    print(f"   ✓ Embedding creado: {embedding.shape}")
    print(f"   ✓ Rango valores: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    # Test 3: Crear DataFrame
    print("\n3. Creando DataFrame con embedding...")
    embedding_df = reducer.get_embedding_dataframe(target, target_names)
    print(f"   ✓ DataFrame creado: {embedding_df.shape}")
    print(f"   ✓ Columnas: {list(embedding_df.columns)}")
    print(f"\n   Primeras 5 filas:")
    print(embedding_df.head())
    
    # Test 4: Estadísticas
    print("\n4. Estadísticas de reducción...")
    original_dim = X_scaled.shape[1]
    stats = reducer.get_reduction_stats(original_dim)
    print(f"   ✓ Dimensionalidad original: {stats['original_dim']}D")
    print(f"   ✓ Dimensionalidad reducida: {stats['reduced_dim']}D")
    print(f"   ✓ Reducción: {stats['reduction_percentage']:.1f}%")
    print(f"   ✓ Número de muestras: {stats['n_samples']}")
    
    # Test 5: Reductor 3D
    print("\n5. Probando reductor 3D...")
    reducer_3d = UMAPReducer()
    reducer_3d.create_reducer(n_components=3, random_state=42)
    embedding_3d = reducer_3d.fit_transform(X_scaled)
    print(f"   ✓ Embedding 3D creado: {embedding_3d.shape}")
    
    return embedding_df

def main():
    """Función principal."""
    print("\n" + "🚀 INICIANDO PRUEBAS BÁSICAS" + "\n")
    
    try:
        # Probar loader
        df, target, target_names, X_scaled = test_loader()
        
        # Probar reducer
        embedding_df = test_reducer(X_scaled, target, target_names)
        
        print("\n" + "=" * 50)
        print("✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ ERROR EN LAS PRUEBAS")
        print("=" * 50)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

