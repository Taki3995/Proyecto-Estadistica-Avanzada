# Plan de Trabajo: Proyecto Estadística Avanzada
## Etapa 1: Preparación y exploración de datos
- Descargar y revisar el dataset de Kaggle
Corporate Bankruptcy Prediction
- Exploración inicial (EDA)
- Identificar variables disponibles (ratios financieros, indicadores contables, etc.)
- Analizar distribución de la variable objetivo (Bankrupt)
- Detectar valores faltantes, outliers y correlaciones
- Preprocesamiento
- Imputación de valores faltantes
- Normalización o estandarización si es necesario
- Codificación de variables si hay categóricas (aunque este dataset es mayormente numérico)

## Etapa 2: Modelado base y estimación clásica
- Aplicar regresión logística MLE
- Estimar coeficientes
- Evaluar significancia estadística
- Interpretar coeficientes en contexto financiero
- Evaluar desempeño inicial
- Métricas: AUC, precisión, recall, F1-score
- Matriz de confusión

## Etapa 3: Técnicas de shrinkage y regularización- Aplicar Ridge Regression
- Comparar coeficientes con MLE
- Analizar reducción de varianza y estabilidad
- Aplicar James-Stein shrinkage
- En variables altamente correlacionadas
- Evaluar impacto en predicción y sesgo

## Etapa 4: Validación y remuestreo- Bootstrap
- Estimar intervalos de confianza para coeficientes
- Evaluar estabilidad de los modelos
- Validación cruzada (k-fold CV)
- Comparar desempeño entre MLE, Ridge y James-Stein
- Seleccionar el modelo más robusto

## Etapa 5: Comparación y análisis técnico- Comparar modelos
- Evaluar trade-off sesgo-varianza
- Comparar métricas de desempeño
- Justificar elección del modelo final
- Interpretación ingenieril
- Traducir resultados a recomendaciones prácticas
- Identificar factores clave de quiebra
- Proponer alertas o criterios de riesgo para empresas

## Etapa 6: Documentación y presentación- Estructurar informe técnico
- Introducción, pregunta, objetivos
- Metodología clara y justificada
- Resultados con gráficos y tablas
- Conclusiones prácticas
- Preparar presentación académica
- Diapositivas con visualizaciones
- Justificación de cada técnica
- Interpretación clara para audiencia no técnica
