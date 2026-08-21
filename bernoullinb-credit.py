"""
bernoullinb-credit.py
Script que integra la particion del dataset (Entrenamiento y Pruebas)
y la elaboracion del modelo de clasificacion de Bayes Ingenuo utilizando BernoulliNB.

Pasos realizados:
1. Carga del conjunto de datos procesado (data/cleaned_credit_data.csv) o ejecutando el pipeline de limpieza.
2. Particionamiento en conjuntos de Entrenamiento (80%) y Prueba (20%) estratificado.
3. Entrenamiento del clasificador BernoulliNB.
4. Evaluacion cuantitativa del modelo (Exactitud, Matriz de Confusion, Precision, Recall, F1-Score y ROC-AUC).
5. Presentacion limpia de resultados en consola sin emojis.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


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

def train_bernoulli_naive_bayes(X_train, y_train, alpha=1.0):
    """
    Entrena el modelo BernoulliNB especifico para caracteristicas de entrada binarias.
    """
    print("Entrenamiento del Clasificador BernoulliNB\n")

    model = BernoulliNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Realiza las predicciones sobre el conjunto de prueba y genera
    el reporte completo de evaluacion cuantitativa.
    """
    print("Evaluacion del Modelo sobre el Conjunto de Prueba\n")

    # Predicciones de clase y probabilidades
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metricas principales
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

def main():
    print("MODELO DE CLASIFICACION CREDITICIA - BERNOULLI NAIVE BAYES\n")
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=42)
    model = train_bernoulli_naive_bayes(X_train, y_train, alpha=1.0)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
