# PREPROCESAMIENTO DEL DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# CARGA DE DATOS

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    print(f"Shape: {df.shape}")
    print("Primeras filas:")
    print(df.head())
    return df

# EXPLORACION INICIAL

def exploracion_inicial(df):
    print("\nColumnas disponibles:")
    print(df.columns.tolist())

    print("\nDistribución de la variable objetivo:")
    print(df['Bankrupt?'].value_counts(normalize=True))

    sns.countplot(x='Bankrupt?', data=df)
    plt.title("Distribución de empresas quebradas vs no quebradas")
    plt.show()


# DETECTAR VALORES FALTANTES

def revisar_nulos(df):
    nulos = df.isnull().mean() * 100
    print("\n% de valores nulos por columna (solo >0%):")
    print(nulos[nulos > 0].sort_values(ascending=False))


# ESTADISTICAS BASICAS Y OUTLIERS

def resumen_estadistico(df):
    print("\nResumen estadístico global:")
    print(df.describe().T)

    # Detección de outliers usando IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print("\nNúmero de outliers por columna (top 10):")
    print(outliers.sort_values(ascending=False).head(10))

# CORRELACIONES

def correlaciones_fuertes(df, threshold=0.8):
    corr = df.corr().abs()
    altas = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index())
    altas.columns = ['Var1','Var2','Corr']
    print(f"\nPares de variables con correlación > {threshold}:")
    print(altas[altas['Corr'] > threshold].sort_values(by='Corr', ascending=False))

# PIPELINE DE PREPROCESAMIENTO

def preprocesar_dataset(df):
    # 1. Imputación con mediana
    imputer = SimpleImputer(strategy='median')
    X = df.drop(columns=['Bankrupt?'])
    y = df['Bankrupt?']

    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 2. Normalización
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    return X_scaled, y, imputer, scaler


# PIPELINE COMPLETO

def pipeline(ruta_csv):
    df = cargar_datos(ruta_csv)
    exploracion_inicial(df)
    revisar_nulos(df)
    resumen_estadistico(df)
    correlaciones_fuertes(df)
    
    X, y, imputer, scaler = preprocesar_dataset(df)
    print("\nDataset preprocesado listo.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y, imputer, scaler



if __name__ == "__main__":
    ruta = "data.csv"
    X, y, imputer, scaler = pipeline(ruta)