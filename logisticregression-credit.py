"""
logisticregression-credit.py
Script que integra la particion del dataset (Entrenamiento y Pruebas)
y la elaboracion de un modelo de regresion logistica para clasificacion de riesgo crediticio.

Pasos realizados:
1. Carga del conjunto de datos procesado (data/cleaned_credit_data.csv) o ejecutando el pipeline de limpieza.
2. Particionamiento en conjuntos de Entrenamiento (80%) y Prueba (20%) estratificado.
3. Entrenamiento del clasificador LogisticRegression.
4. Evaluacion cuantitativa del modelo (Exactitud, Matriz de Confusion, Precision, Recall, F1-Score y ROC-AUC).
5. Presentacion limpia de resultados en consola sin emojis.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# Carga del dataset ya limpio. Si no existe, se ejecuta primero el pipeline de limpieza.
def load_processed_data():
    """Carga los datos limpios de data/cleaned_credit_data.csv o ejecuta la limpieza si no existe."""
    filepath = os.path.join("data", "cleaned_credit_data.csv")
    if not os.path.exists(filepath):
        print("Dataset limpio no encontrado. Ejecutando script de limpieza clean_data.py...\n")
        import clean_data
        clean_data.main()

    df = pd.read_csv(filepath)
    X = df.drop(columns=['target_credit_risk'])
    y = df['target_credit_risk']
    return X, y


# Particionamiento final del conjunto para evaluar el modelo sobre datos no vistos.
def split_dataset(X, y, test_size=0.20, random_state=42):
    """
    Realiza el particionamiento entre conjuntos de entrenamiento y prueba.
    Se utiliza estratificacion (stratify=y) para preservar la proporcion
    70% Buen Credito y 30% Mal Credito en ambos subconjuntos.
    """
    print("Particionando el Dataset (Train/Test Split)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# Eleccion del modelo: Regresion Logistica.
# Se selecciona porque el dataset ya fue binarizado y codificado, por lo que la regresion logistica
# suele funcionar mejor que Naive Bayes cuando las variables tienen relacion lineal y no solo independencia.
# Ademas, permite un mejor equilibrio entre sensibilidad y especificidad con class_weight='balanced'.
def train_logistic_regression(X_train, y_train, max_iter=1000, random_state=42):
    """
    Entrena el modelo de Regresion Logistica especifico para datos binarios y variables transformadas.
    """
    print("Entrenamiento del Clasificador Logistic Regression\n")

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight='balanced',
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    return model


# Evaluacion del modelo sobre el conjunto de prueba.
# Se imprimen exactamente las mismas metricas que en el script BernoulliNB para comparar resultados.
def evaluate_model(model, X_test, y_test):
    """
    Realiza las predicciones sobre el conjunto de prueba y genera
    el reporte completo de evaluacion cuantitativa.
    """
    print("Evaluacion del Modelo sobre el Conjunto de Prueba\n")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f"Exactitud Global (Accuracy): {acc * 100:.2f}%")
    print(f"Area Bajo la Curva ROC (ROC-AUC): {roc_auc:.4f}")

    print("\nMatriz de Confusion:")
    print(f"                  Predicho Buen Credito (0)   Predicho Mal Credito (1)")
    print(f"Real Buen Credito (0)      {tn:<27} {fp}")
    print(f"Real Mal Credito (1)       {fn:<27} {tp}")

    print("\nDetalle de Desglose de Predicciones:")
    print(f"  - Verdaderos Negativos (TN - Buen Credito correcto): {tn}")
    print(f"  - Falsos Positivos     (FP - Buen Credito clasificado como Malo): {fp}")
    print(f"  - Falsos Negativos     (FN - Mal Credito no detectado): {fn}")
    print(f"  - Verdaderos Positivos (TP - Mal Credito detectado correctamente): {tp}")

    print("\nReporte Metrico de Clasificacion por Clase:")
    target_names = ['Buen Credito (0)', 'Mal Credito (1)']
    print(classification_report(y_test, y_pred, target_names=target_names))


# Validacion cruzada opcional para depurar el modelo y comprobar estabilidad del rendimiento.
def cross_validate_model(X, y, n_splits=5, random_state=42):
    """Evalua la regresion logistica con validacion cruzada estratificada."""
    print("Validacion Cruzada Estratificada (5 folds)\n")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced', solver='liblinear')

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print("Accuracy por fold:", scores)
    print(f"Accuracy media: {scores.mean():.4f}")
    print(f"Accuracy std: {scores.std():4f}\n")


# Funcion principal que ejecuta el flujo completo del modelo.
def main():
    print("MODELO DE CLASIFICACION CREDITICIA - REGRESION LOGISTICA\n")
    X, y = load_processed_data()

    # Se ejecuta validacion cruzada para depurar la estabilidad del modelo antes del train/test final.
    cross_validate_model(X, y, n_splits=5, random_state=42)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=42)
    model = train_logistic_regression(X_train, y_train, max_iter=1000, random_state=42)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
