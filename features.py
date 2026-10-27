"""Engenharia de features: transforma pedidos brutos em uma linha por cliente.

Ideia central: escolhemos uma data de referência (`reference_date`).
- Tudo que acontece ANTES dela pode virar feature (passado).
- O que acontece DEPOIS dela é usado só para criar o rótulo (futuro).

Isso evita "vazamento de dados": o modelo nunca enxerga o futuro na hora de treinar.
"""

from __future__ import annotations

import pandas as pd

from food_intelligence import config

# Ordem fixa das features. A API usa esta mesma lista para validar a entrada.
FEATURE_COLUMNS = [
    "recency_days",      # dias desde o último pedido (quanto maior, pior)
    "frequency_90d",     # quantidade de pedidos na janela de observação
    "monetary_90d",      # faturamento gerado pelo cliente na janela (R$)
    "avg_order_value",   # ticket médio por pedido (R$)
    "n_categories",      # quantas categorias diferentes o cliente comprou
    "tenure_days",       # há quantos dias o cliente compra da distribuidora
]


def build_customer_features(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    reference_date: pd.Timestamp,
    observation_days: int = config.OBSERVATION_DAYS,
) -> pd.DataFrame:
    """Calcula as features de cada cliente que comprou na janela de observação.

    Retorna um DataFrame indexado por `customer_id`, com as colunas de FEATURE_COLUMNS.
    """
    ref = pd.Timestamp(reference_date)
    window_start = ref - pd.Timedelta(days=observation_days)

    # Só o passado: pedidos anteriores à data de referência, com categoria e valor do item
    past = orders[orders["order_date"] < ref].merge(
        products[["product_id", "category"]], on="product_id", how="left"
    )
    past = past.assign(line_total=past["quantity"] * past["unit_price"])

    # Janela de observação: últimos `observation_days` antes da referência
    window = past[past["order_date"] >= window_start]
    grouped = window.groupby("customer_id")

    features = pd.DataFrame(
        {
            "frequency_90d": grouped["order_id"].nunique(),
            "monetary_90d": grouped["line_total"].sum(),
            "n_categories": grouped["category"].nunique(),
            "last_order": grouped["order_date"].max(),
        }
    )
    features["avg_order_value"] = features["monetary_90d"] / features["frequency_90d"]
    features["recency_days"] = (ref - features["last_order"]).dt.days

    # Relacionamento: conta desde o primeiro pedido de TODA a história (não só a janela)
    first_order = past.groupby("customer_id")["order_date"].min()
    features["tenure_days"] = (ref - first_order.reindex(features.index)).dt.days

    return features[FEATURE_COLUMNS]


def build_churn_labels(
    orders: pd.DataFrame,
    customer_ids: pd.Index,
    reference_date: pd.Timestamp,
    churn_window_days: int = config.CHURN_WINDOW_DAYS,
) -> pd.Series:
    """Rótulo: 1 = churn (não comprou na janela futura), 0 = continuou comprando.

    A janela futura começa na data de referência e dura `churn_window_days` dias.
    """
    ref = pd.Timestamp(reference_date)
    end = ref + pd.Timedelta(days=churn_window_days)

    future = orders[(orders["order_date"] >= ref) & (orders["order_date"] < end)]
    bought_again = pd.Index(customer_ids).isin(future["customer_id"].unique())

    return pd.Series((~bought_again).astype(int), index=customer_ids, name="churn")
