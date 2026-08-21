"""
clean_data.py
Script para la limpieza, renombramiento explicito y preparacion del conjunto de datos
Statlog (German Credit Data) para el modelo de clasificacion de Bayes Ingenuo (BernoulliNB).

Este script realiza:
1. Carga del conjunto de datos mediante ucimlrepo (id=144).
2. Renombramiento explicito de las variables Attribute1 a Attribute20 a nombres descriptivos.
3. Justificacion de la seleccion de variables para el modelo.
4. Mapeo de la variable objetivo binaria (class).
5. Codificacion One-Hot Encoding para variables categoricas y binarizacion de variables continuas.
6. Guardado del dataset procesado en data/cleaned_credit_data.csv.
"""

import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo

def load_raw_data():
    """Carga el dataset Statlog German Credit Data desde UCI Repository."""
    print("Cargando Datos desde UCI Repository\n")
    print("Descargando dataset Statlog (German Credit Data) ID: 144...\n")
    statlog = fetch_ucirepo(id=144)

    X = statlog.data.features.copy()
    y = statlog.data.targets.copy()

    return X, y

def rename_variables(X, y):
    """
    Renombra las variables de Attribute1..Attribute20 a nombres significativos
    basados en el diccionario de datos del problema.
    """
    print("Renombrando Variables...\n")

    mapping = {
        'Attribute1': 'status_checking_account',      # Status of existing checking account
        'Attribute2': 'duration_months',              # Duration (months)
        'Attribute3': 'credit_history',               # Credit history
        'Attribute4': 'purpose',                      # Purpose
        'Attribute5': 'credit_amount',                # Credit amount
        'Attribute6': 'savings_account_bonds',        # Savings account/bonds
        'Attribute7': 'present_employment_since',    # Present employment since
        'Attribute8': 'installment_rate',             # Installment rate in % of disposable income
        'Attribute9': 'personal_status_sex',          # Personal status and sex
        'Attribute10': 'other_debtors_guarantors',    # Other debtors / guarantors
        'Attribute11': 'present_residence_since',     # Present residence since
        'Attribute12': 'property',                    # Property
        'Attribute13': 'age_years',                   # Age (years)
        'Attribute14': 'other_installment_plans',     # Other installment plans
        'Attribute15': 'housing',                     # Housing
        'Attribute16': 'number_existing_credits',     # Number of existing credits at this bank
        'Attribute17': 'job',                         # Occupation / Job
        'Attribute18': 'number_people_liable',        # Number of people being liable
        'Attribute19': 'telephone',                   # Telephone
        'Attribute20': 'foreign_worker'               # Foreign worker
    }

    X_renamed = X.rename(columns=mapping)
    y_renamed = y.rename(columns={'class': 'credit_risk'})

    return X_renamed, y_renamed

def transform_and_clean_data(X, y):
    """
    Transforma la variable objetivo y codifica las caracteristicas a formato binario.
    Target: 1 (Buen Credito) -> 0, 2 (Mal Credito) -> 1 (Donde 1 representa evento de riesgo).
    Categoricas: One-Hot Encoding (variables binarias 0/1).
    Numericas: Binarizadas con respecto a la mediana (> mediana = 1, <= mediana = 0).
    """
    print("Limpiando, Transformando y Binarizando\n")

    # 4.1 Transformacion del Target
    # Originalmente 1 = Good, 2 = Bad
    y_binary = (y['credit_risk'] == 2).astype(int)

    # 4.2 Separacion de variables categoricas y numericas
    cat_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 4.3 One-Hot Encoding para variables categoricas
    X_cat_encoded = pd.get_dummies(X[cat_cols], drop_first=False, dtype=int)

    # 4.4 Binarizacion de variables numericas (mayor a la mediana = 1, menor o igual = 0)
    X_num_binarized = pd.DataFrame()
    for col in num_cols:
        median_val = X[col].median()
        col_name = f"{col}_above_median"
        X_num_binarized[col_name] = (X[col] > median_val).astype(int)

    # 4.5 Concatenacion de matrices procesadas
    X_clean = pd.concat([X_cat_encoded, X_num_binarized], axis=1)

    return X_clean, y_binary

def save_cleaned_dataset(X_clean, y_binary):
    """Guarda el conjunto de datos limpio y procesado en la carpeta data/."""
    print("Exportando el dataset limpio...\n")

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cleaned_credit_data.csv")

    # Combinar X e y para la exportacion CSV
    df_clean = X_clean.copy()
    df_clean['target_credit_risk'] = y_binary

    df_clean.to_csv(output_path, index=False)
    print(f"Dataset limpio guardado en: {output_path}\n")

def main():
    print("LIMPIEZA DE DATOS - GERMAN CREDIT DATA\n")
    X_raw, y_raw = load_raw_data()
    X_renamed, y_renamed = rename_variables(X_raw, y_raw)

    # DataFrame consolidado para analisis de nulos y justificacion
    df_temp = pd.concat([X_renamed, y_renamed], axis=1)

    X_clean, y_binary = transform_and_clean_data(X_renamed, y_renamed)
    save_cleaned_dataset(X_clean, y_binary)

if __name__ == "__main__":
    main()
