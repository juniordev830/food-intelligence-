"""Módulo 3 — Elasticidade-preço: quanto a demanda reage ao preço?

Modelo (regressão log-log, uma por produto):

    log(quantidade) = a + e * log(preço)

O coeficiente `e` é a elasticidade:
- e = -2.0  -> preço sobe 10%, quantidade cai ~20% (demanda elástica)
- e = -0.5  -> preço sobe 10%, quantidade cai ~5%  (demanda inelástica)

Limitação importante: isto mede correlação em dados observacionais. Se o preço
subiu justamente em meses de alta demanda, a estimativa fica enviesada. Em
produção, o ideal é combinar com testes A/B de preço.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MIN_OBSERVATIONS = 10  # menos que isso não gera estimativa confiável


def estimate_elasticities(orders: pd.DataFrame) -> pd.DataFrame:
    """Estima a elasticidade de cada produto. Retorna um DataFrame indexado por product_id."""
    rows = []
    for product_id, group in orders.groupby("product_id"):
        if len(group) < MIN_OBSERVATIONS:
            continue

        # Transformação logarítmica: o coeficiente da regressão vira a elasticidade
        X = np.log(group[["unit_price"]].to_numpy())
        y = np.log(group["quantity"].to_numpy())

        reg = LinearRegression().fit(X, y)
        rows.append(
            {
                "product_id": product_id,
                "elasticity": round(float(reg.coef_[0]), 4),
                "r2": round(float(reg.score(X, y)), 4),  # quanto do comportamento o preço explica
                "n_obs": int(len(group)),
            }
        )

    return pd.DataFrame(rows).set_index("product_id")


def simulate_price_change(
    elasticity: float,
    current_price: float,
    new_price: float,
    current_quantity: float,
) -> dict:
    """Prevê a quantidade e a receita após uma mudança de preço.

    Usa a forma log-log: nova_qtd = qtd_atual * (novo_preço / preço_atual) ** e
    """
    ratio = new_price / current_price
    new_quantity = current_quantity * ratio**elasticity

    revenue_before = current_price * current_quantity
    revenue_after = new_price * new_quantity

    return {
        "predicted_quantity": round(float(new_quantity), 2),
        "revenue_change_pct": round(float((revenue_after / revenue_before - 1) * 100), 2),
        # Demanda elástica: a quantidade muda mais que proporcionalmente ao preço
        "classification": "elástica" if abs(elasticity) > 1 else "inelástica",
    }
