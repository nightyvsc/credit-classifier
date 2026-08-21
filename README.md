# Clasificacion de Riesgo Crediticio con Bernoulli Naive Bayes

Analisis de clasificacion sobre el conjunto de datos Statlog (German Credit Data) de UCI Machine Learning Repository (ID: 144) para predecir la condicion de riesgo de credito.

## Estructura del Proyecto

- `eda.ipynb`: Jupyter Notebook con la exploracion de datos, distribuciones y matriz de correlacion.
- `clean_data.py`: Script de limpieza, renombramiento explicito de variables a nombres descriptivos y binarizacion de caracteristicas.
- `bernoullinb-credit.py`: Script de particionamiento (80/20 estratificado), entrenamiento y evaluacion del clasificador BernoulliNB.
- `requirements.txt`: Lista de librerias de Python requeridas.
- `data/cleaned_credit_data.csv`: Archivo generado por `clean_data.py` con los datos procesados.

## Instalacion y Entorno Virtual

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Guia de Ejecucion del Pipeline

### 1. Limpieza y Transformacion de Datos
Ejecuta el script para procesar el dataset desde UCI, renombrar atributos a nombres descriptivos, codificar variables categoricas mediante One-Hot Encoding y binarizar caracteristicas numericas:

```bash
python clean_data.py
```

### 2. Entrenamiento y Evaluacion del Modelo
Ejecuta el script para realizar la particion entre conjuntos de entrenamiento y prueba (80% train / 20% test con estratificacion), entrenar el clasificador `BernoulliNB` y desplegar las metricas de evaluacion (Exactitud, Matriz de Confusion, Reporte de Clasificacion y ROC-AUC):

```bash
python bernoullinb-credit.py
```
