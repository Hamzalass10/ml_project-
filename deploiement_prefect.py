"""
deploiement_prefect.py
Enregistre des déploiements Prefect pour les flows du pipeline ML.

Chaque déploiement :
- pointe vers un flow de pipeline_prefect.py
- est planifié (cron) pour s'exécuter automatiquement
- peut aussi être déclenché manuellement via `prefect deployment run`

Prérequis :
    Terminal 1 : prefect server start
    Terminal 2 : python deploiement_prefect.py serve
    Terminal 3 : prefect deployment run 'all/ml-pipeline-all'
"""

import argparse
from prefect import serve

from pipeline_prefect import (
    all_flow,
    train_flow,
    evaluate_flow,
    code_flow,
)


def build_deployments():
    """Construit la liste des déploiements à enregistrer."""
    # Chaque tuple = (flow, nom, cron, tags)
    specs = [
        (all_flow,      "ml-pipeline-all",      "0 2 * * *", ["full-pipeline", "mlops"]),
        (train_flow,    "ml-pipeline-train",    "0 3 * * *", ["training", "mlops"]),
        (evaluate_flow, "ml-pipeline-evaluate", "0 4 * * *", ["evaluation", "mlops"]),
        (code_flow,     "ml-pipeline-code",     "0 1 * * *", ["quality", "mlops"]),
    ]

    deployments = []
    for flow_obj, name, cron, tags in specs:
        dep = flow_obj.to_deployment(
            name=name,
            cron=cron,
            tags=tags,
            description=f"Déploiement planifié : {name}",
        )
        deployments.append(dep)
    return deployments


def main():
    parser = argparse.ArgumentParser(description="Déploiement Prefect - CLI")
    parser.add_argument(
        "command",
        choices=["serve"],
        help="Commande à exécuter (serve = démarre le worker)",
    )
    args = parser.parse_args()

    deployments = build_deployments()

    print(f"🚀 Enregistrement de {len(deployments)} déploiement(s)...")
    for dep in deployments:
        print(f"   - {dep.name}")

    # `serve` démarre un worker local qui exécute les flows planifiés
    serve(*deployments)


if __name__ == "__main__":
    main()
