"""
model_pipeline.py
Fonctions modulaires du pipeline ML pour la prédiction du churn client.
Basé sur customer_churn.ipynb (43 cellules).
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# 1. Préparation des données
# ---------------------------------------------------------------------------
def prepare_data(
    csv_path: str = "Churn_Modelling.csv",
    target_column: str = "Exited",
    test_size: float = 0.2,
    random_state: int = 1,
):
    """
    Charge le CSV, nettoie, encode, split et normalise les données.

    Retourne :
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # --- Chargement ---
    df = pd.read_csv(csv_path)

    # --- Nettoyage (cellules 20-29 du notebook) ---
    # Encodage de la variable catégorielle Gender
    if "Gender" in df.columns:
        df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    # Suppression des colonnes inutiles
    columns_to_drop = ["Surname", "Geography", "RowNumber", "CustomerId"]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    # --- Séparation features / cible ---
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- Normalisation (CORRIGÉE : fit sur train, transform sur test) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, list(X.columns)


# ---------------------------------------------------------------------------
# 2. Entraînement
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Entraîne un RandomForestClassifier (cellules 33-34 du notebook).
    Retourne le modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 3. Évaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, verbose: bool = True):
    """
    Évalue le modèle (cellules 36-41 du notebook).
    Retourne un dict avec accuracy, rapport de classification et matrice.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    if verbose:
        print(f"Accuracy : {acc:.4f}")
        print(classification_report(y_test, y_pred))
        print("Matrice de confusion :\n", cm)

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


# ---------------------------------------------------------------------------
# 4. Sauvegarde
# ---------------------------------------------------------------------------
def save_model(model, scaler, output_dir: str = "artifacts"):
    """
    Sauvegarde le modèle et le scaler avec joblib (cellule 43 du notebook).
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Modèle sauvegardé : {model_path}")
    print(f"Scaler sauvegardé : {scaler_path}")
    return model_path, scaler_path


# ---------------------------------------------------------------------------
# 5. Chargement
# ---------------------------------------------------------------------------
def load_model(
    model_path: str = "artifacts/model.joblib",
    scaler_path: str = "artifacts/scaler.joblib",
):
    """
    Charge le modèle et le scaler depuis le disque.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Modèle chargé depuis {model_path}")
    return model, scaler


# ---------------------------------------------------------------------------
# 6. Prédiction
# ---------------------------------------------------------------------------
def predict(model, scaler, X_new):
    """
    Applique le pipeline de prédiction sur de nouvelles données (cellule 39).
    X_new : DataFrame ou array-like avec les mêmes colonnes que X_train.
    """
    X_scaled = scaler.transform(X_new)
    return model.predict(X_scaled)
