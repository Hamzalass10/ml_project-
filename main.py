"""
main.py
Point d'entrée CLI pour exécuter les étapes du pipeline ML.
"""

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
)

CSV_PATH = "Churn_Modelling.csv"


def cmd_prepare(args):
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(CSV_PATH)
    print(f"Données prêtes : {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(f"Features : {features}")


def cmd_train(args):
    X_train, X_test, y_train, y_test, scaler, _ = prepare_data(CSV_PATH)
    model = train_model(X_train, y_train)
    save_model(model, scaler)
    print("Entraînement terminé et modèle sauvegardé.")


def cmd_evaluate(args):
    X_train, X_test, y_train, y_test, scaler, _ = prepare_data(CSV_PATH)
    model, _ = load_model()
    evaluate_model(model, X_test, y_test)


def cmd_predict(args):
    model, scaler = load_model()
    # Exemple : prédire sur 5 premières lignes du CSV
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(CSV_PATH)
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df = df.drop(columns=["Surname", "Geography", "RowNumber", "CustomerId", "Exited"])
    preds = predict(model, scaler, df.head(5))
    print("Prédictions sur 5 clients :", preds)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML Churn - CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    sub.add_parser("predict").set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
