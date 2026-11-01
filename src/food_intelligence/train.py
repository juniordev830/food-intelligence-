"""Pipeline de treinamento: dados -> features -> modelos -> artefatos.

Uso:
    python -m food_intelligence.train
    python -m food_intelligence.train --customers 1200 --seed 7

Ao final, a pasta `artifacts/` terá:
- churn_model.joblib   (modelo de churn)
- recommender.joblib   (matriz de similaridade entre produtos)
- elasticity.joblib    (elasticidade estimada por produto)
- metrics.json         (métricas para registrar o resultado do treino)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from food_intelligence import config
from food_intelligence.churn import train_churn_model
from food_intelligence.data_generator import generate_dataset, load_tables, save_tables
from food_intelligence.elasticity import estimate_elasticities
from food_intelligence.features import build_churn_labels, build_customer_features
from food_intelligence.recommender import build_similarity


def reference_date() -> pd.Timestamp:
    """Data de corte entre passado (features) e futuro (rótulo de churn).

    Escolhemos a data de forma que a janela futura caiba inteira dentro da simulação.
    """
    start = pd.Timestamp(config.SIMULATION_START)
    return start + pd.Timedelta(days=config.SIMULATION_DAYS - config.CHURN_WINDOW_DAYS)


def run_pipeline(
    n_customers: int = 800,
    seed: int = config.RANDOM_SEED,
    data_dir: Path = config.DATA_DIR,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
) -> dict:
    """Executa o treino completo e devolve as métricas."""

    # 1) Dados: gera, salva em CSV e relê (simula a ingestão de dados reais)
    print("[1/5] Gerando dados sintéticos...")
    tables = generate_dataset(n_customers=n_customers, seed=seed)
    save_tables(tables, data_dir)
    data = load_tables(data_dir)
    orders, products = data["orders"], data["products"]

    # 2) Churn: features do passado + rótulo do futuro
    print("[2/5] Criando features e rótulo de churn...")
    ref = reference_date()
    X = build_customer_features(orders, products, ref)
    y = build_churn_labels(orders, X.index, ref)

    print("[3/5] Treinando modelo de churn...")
    churn_model, churn_metrics = train_churn_model(X, y, seed)

    # 3) Recomendação: matriz de similaridade entre produtos
    print("[4/5] Calculando similaridade entre produtos...")
    similarity = build_similarity(orders)

    # 4) Elasticidade: uma regressão por produto
    print("[5/5] Estimando elasticidade-preço...")
    elasticities = estimate_elasticities(orders)

    # 5) Salva artefatos para a API carregar depois
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(churn_model, artifacts_dir / "churn_model.joblib")
    joblib.dump(similarity, artifacts_dir / "recommender.joblib")
    joblib.dump(elasticities, artifacts_dir / "elasticity.joblib")

    metrics = {
        "churn": churn_metrics,
        "reference_date": str(ref.date()),
        "n_orders": int(len(orders)),
        "n_products": int(len(products)),
        "elasticity_mean": round(float(elasticities["elasticity"].mean()), 4),
    }
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina os modelos do Food Intelligence.")
    parser.add_argument("--customers", type=int, default=800, help="número de clientes sintéticos")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="semente aleatória")
    args = parser.parse_args()

    metrics = run_pipeline(n_customers=args.customers, seed=args.seed)
    print("\nTreino concluído! Métricas:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
