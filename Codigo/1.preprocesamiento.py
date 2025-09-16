# PREPROCESAMIENTO DEL DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler #para proteger de outliers
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize #para limitar outliers


# CARGA DE DATOS

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df.columns = df.columns.str.strip()  # limpiar espacios
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

def correlaciones_fuertes(df, threshold=0.95):
    corr = df.corr().abs()
    altas = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index())
    altas.columns = ['Var1','Var2','Corr']
    altas_filtradas = altas[altas['Corr'] > threshold].sort_values(by='Corr', ascending=False)
    return altas_filtradas

# WINSORIZACION (control de outliers)

def winsorizar_df(df, limits=[0.01, 0.01]):
    df_w = df.copy()
    for col in df_w.drop(columns=['Bankrupt?']).columns:
        df_w[col] = winsorize(df_w[col], limits=limits)
    return df_w

# ELIMINAR CONSTANTES Y REDUNDANTES

def limpiar_columnas(df, corr_pairs):
    # Eliminar columnas constantes
    constantes = [col for col in df.columns if df[col].nunique() == 1]

    # Eliminar una de cada par altamente correlacionado
    redundantes = set()
    for _, row in corr_pairs.iterrows():
        redundantes.add(row['Var2'])  # nos quedamos con Var1 arbitrariamente

    eliminar = list(set(constantes) | redundantes)
    print(f"\nColumnas eliminadas automáticamente ({len(eliminar)}):")
    print(eliminar)

    return df.drop(columns=eliminar, errors="ignore")


# TOP 15 VARIABLES CLAVE 

def variables_prioritarias():
    return [
        'ROA(C) before interest and depreciation before interest',
        'ROA(A) before interest and % after tax',
        'Operating Gross Margin',
        'Net Income to Total Assets',
        'Debt ratio %',
        'Net worth/Assets',
        'Liability to Equity',
        'Cash Flow to Total Assets',
        'Cash Flow to Liability',
        'Current Ratio',
        'Quick Ratio',
        'Working Capital to Total Assets',
        'Interest Coverage Ratio (Interest expense to EBIT)',
        'Operating Profit Growth Rate',
        'Gross Profit to Sales'
    ]

# PIPELINE DE PREPROCESAMIENTO

def preprocesar_dataset(df):
     # 1. Winsorización
    df_w = winsorizar_df(df)

    # 2. Reportar y eliminar redundantes
    corr_pairs = correlaciones_fuertes(df_w)
    df_clean = limpiar_columnas(df_w, corr_pairs)

    # 3. Imputación
    imputer = SimpleImputer(strategy='median')
    X = df_clean.drop(columns=['Bankrupt?'])
    y = df_clean['Bankrupt?']
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 4. Escalado robusto
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    return X_scaled, y, imputer, scaler

def graficar_boxplots(X, top_n=15):
    # Elegimos las variables con mayor importancia/variabilidad
    # Para simplicidad, usamos la desviación estándar como proxy
    stds = X.std().sort_values(ascending=False)
    top_vars = stds.head(top_n).index.tolist()
    
    plt.figure(figsize=(15,7))
    sns.boxplot(data=X[top_vars], orient='h')
    plt.title(f"Boxplots de las {top_n} variables más relevantes (mayor variabilidad)")
    plt.xlabel("Valor estandarizado")
    plt.ylabel("Variables")
    plt.show()
    
    print("\nVariables incluidas en el boxplot:")
    print(top_vars)

def graficar_correlacion(X):
    plt.figure(figsize=(12,10))
    sns.heatmap(X.corr(), cmap='coolwarm', center=0, annot=False)
    plt.title("Correlación entre variables preprocesadas")
    plt.show()

# PIPELINE COMPLETO

def pipeline(ruta_csv):
    df = cargar_datos(ruta_csv)
    exploracion_inicial(df)
    revisar_nulos(df)
    resumen_estadistico(df)

    X, y, imputer, scaler = preprocesar_dataset(df)
    print("\nDataset preprocesado listo.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    print("\nVariables prioritarias sugeridas (Top 15):")
    print(variables_prioritarias())

    graficar_boxplots(X, top_n=15)
    graficar_correlacion(X)

    return X, y, imputer, scaler




if __name__ == "__main__":
    ruta = "data.csv"
    X, y, imputer, scaler = pipeline(ruta)