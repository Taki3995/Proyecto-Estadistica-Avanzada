# PREPROCESAMIENTO DEL DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# CARGAR DATASET Y MOSTRAR INFORMACIÓN BÁSICA

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
