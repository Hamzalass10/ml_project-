"""
test_model_pipeline.py
Tests unitaires du pipeline ML.
Exécution : pytest test_model_pipeline.py -v
"""

import os
import numpy as np
import pytest

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
)

CSV_PATH = "Churn_Modelling.csv"
ARTIFACTS_DIR = "artifacts_test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prepared_data():
    """Prépare les données une seule fois pour tous les tests."""
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(CSV_PATH)
    return X_train, X_test, y_train, y_test, scaler, features


@pytest.fixture(scope="module")
def trained_model(prepared_data):
    """Entraîne un modèle une seule fois pour tous les tests."""
    X_train, X_test, y_train, y_test, scaler, _ = prepared_data
    return train_model(X_train, y_train)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_prepare_data_shapes(prepared_data):
    """Vérifie que le split et les dimensions sont corrects."""
    X_train, X_test, y_train, y_test, scaler, features = prepared_data
    assert X_train.shape[0] == 8000, "Le train doit contenir 8000 échantillons"
    assert X_test.shape[0] == 2000, "Le test doit contenir 2000 échantillons"
    assert len(features) == 9, "Il doit y avoir 9 features"
    assert X_train.shape[1] == len(features)


def test_prepare_data_no_nan(prepared_data):
    """Vérifie qu'il n'y a aucun NaN après préparation."""
    X_train, X_test, y_train, y_test, scaler, _ = prepared_data
    assert not np.isnan(X_train).any(), "X_train ne doit pas contenir de NaN"
    assert not np.isnan(X_test).any(), "X_test ne doit pas contenir de NaN"


def test_train_model_returns_fitted_model(trained_model):
    """Vérifie que le modèle est entraîné et prédit correctement."""
    assert hasattr(trained_model, "predict"), "Le modèle doit avoir une méthode predict"
    # Le modèle doit être capable de faire des prédictions
    assert trained_model.n_features_in_ == 9


def test_evaluate_model_accuracy(trained_model, prepared_data):
    """Vérifie que l'accuracy est au-dessus d'un seuil raisonnable."""
    X_train, X_test, y_train, y_test, scaler, _ = prepared_data
    metrics = evaluate_model(trained_model, X_test, y_test, verbose=False)
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.75, "L'accuracy doit être > 0.75"
    assert metrics["accuracy"] < 1.0, "L'accuracy ne doit pas être parfaite"


def test_save_and_load_model(trained_model, prepared_data):
    """Vérifie le cycle sauvegarde → chargement."""
    _, _, _, _, scaler, _ = prepared_data
    model_path, scaler_path = save_model(trained_model, scaler, output_dir=ARTIFACTS_DIR)

    assert os.path.exists(model_path), "Le fichier modèle doit exister"
    assert os.path.exists(scaler_path), "Le fichier scaler doit exister"

    loaded_model, loaded_scaler = load_model(model_path, scaler_path)
    assert loaded_model is not None
    assert loaded_scaler is not None


def test_predict_returns_binary(trained_model, prepared_data):
    """Vérifie que les prédictions sont binaires (0 ou 1)."""
    X_train, X_test, y_train, y_test, scaler, _ = prepared_data
    preds = predict(trained_model, scaler, X_test[:10])
    assert len(preds) == 10, "Doit retourner 10 prédictions"
    assert set(np.unique(preds)).issubset({0, 1}), "Prédictions doivent être 0 ou 1"


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------
def teardown_module(module):
    """Supprime le dossier artifacts_test après les tests."""
    import shutil
    if os.path.exists(ARTIFACTS_DIR):
        shutil.rmtree(ARTIFACTS_DIR)
