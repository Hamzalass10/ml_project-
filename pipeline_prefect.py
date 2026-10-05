"""
pipeline_prefect.py
Orchestration du pipeline ML avec Prefect.

Tasks  : install, format, quality, security, unit_tests,
         prepare_data, train_model, save_model, load_model, evaluate_model
Flows  : code, all, train, evaluate

Usage :
    python pipeline_prefect.py --flow code
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
"""

import argparse
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)

CSV_PATH = "Churn_Modelling.csv"

# Fichiers ciblés pour les étapes qualité / sécurité
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


# ---------------------------------------------------------------------------
# TASKS — Environnement
# ---------------------------------------------------------------------------
@task(name="install-dependencies", log_prints=True)
def install_dependencies():
    """Installe les dépendances listées dans requirements.txt."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )


# ---------------------------------------------------------------------------
# TASKS — Qualité du code
# ---------------------------------------------------------------------------
@task(name="format-code", log_prints=True)
def format_code():
    """Formate le code avec black."""
    subprocess.run(
        [sys.executable, "-m", "black", *CODE_FILES],
        check=True,
    )


@task(name="check-quality", log_prints=True)
def check_quality():
    """Vérifie la qualité du code avec pylint (note /10)."""
    result = subprocess.run(
        [sys.executable, "-m", "pylint", *CODE_FILES, "--score=y"],
        capture_output=True,
        text=True,
        check=False,  # on ne bloque pas même si pylint râle
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    # Extraire la note finale si présente
    for line in result.stdout.splitlines():
        if "rated at" in line:
            print(f"👉 {line.strip()}")


@task(name="check-security", log_prints=True)
def check_security():
    """Analyse la sécurité du code avec bandit."""
    result = subprocess.run(
        [sys.executable, "-m", "bandit", "-r", *CODE_FILES],
        capture_output=True,
        text=True,
        check=False,  # on rapporte sans bloquer
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


@task(name="run-unit-tests", log_prints=True)
def run_unit_tests():
    """Exécute les tests unitaires avec pytest."""
    subprocess.run(
        [sys.executable, "-m", "pytest", "test_model_pipeline.py", "-v"],
        check=True,
    )


# ---------------------------------------------------------------------------
# TASKS — Pipeline ML
# ---------------------------------------------------------------------------
@task(name="prepare-data", log_prints=True)
def prepare_data_task():
    """Charge, nettoie et prépare les données."""
    return prepare_data(CSV_PATH)


@task(name="train-model", log_prints=True)
def train_model_task(X_train, y_train):
    """Entraîne le modèle RandomForest."""
    return train_model(X_train, y_train)


@task(name="save-model", log_prints=True)
def save_model_task(model, scaler):
    """Sauvegarde le modèle et le scaler sur disque."""
    return save_model(model, scaler)


@task(name="load-model", log_prints=True)
def load_model_task():
    """Charge le modèle et le scaler depuis le disque."""
    return load_model()


@task(name="evaluate-model", log_prints=True)
def evaluate_model_task(model, X_test, y_test):
    """Évalue le modèle sur le jeu de test."""
    return evaluate_model(model, X_test, y_test)


# ---------------------------------------------------------------------------
# FLOWS
# ---------------------------------------------------------------------------
@flow(name="code", log_prints=True)
def code_flow():
    """Pipeline qualité : format → qualité → sécurité → tests."""
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="all", log_prints=True)
def all_flow():
    """Pipeline complet : install → code → prepare → train → save → evaluate."""
    install_dependencies()
    code_flow()  # ← intégré comme sous-flow
    X_train, X_test, y_train, y_test, scaler, _ = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model, scaler)
    evaluate_model_task(model, X_test, y_test)


@flow(name="train", log_prints=True)
def train_flow():
    """Pipeline d'entraînement : prepare → train → save."""
    X_train, X_test, y_train, y_test, scaler, _ = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model, scaler)


@flow(name="evaluate", log_prints=True)
def evaluate_flow():
    """Pipeline d'évaluation : prepare → load → evaluate."""
    X_train, X_test, y_train, y_test, scaler, _ = prepare_data_task()
    model, _ = load_model_task()
    evaluate_model_task(model, X_test, y_test)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect - CLI")
    parser.add_argument(
        "--flow",
        choices=["code", "all", "train", "evaluate"],
        required=True,
        help="Flow à exécuter",
    )
    args = parser.parse_args()

    if args.flow == "code":
        code_flow()
    elif args.flow == "all":
        all_flow()
    elif args.flow == "train":
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()
