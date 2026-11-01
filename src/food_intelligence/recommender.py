"""Módulo 2 — Recomendação de produtos (filtragem colaborativa item-a-item).

Lógica: produtos que costumam ser comprados pelos MESMOS clientes são parecidos.
Se um cliente compra o produto A, sugerimos os produtos mais parecidos com A
que ele ainda não comprou.

Passos:
1. Monta a matriz cliente x produto (quantidade total comprada).
2. Calcula a similaridade de cosseno entre cada par de produtos (colunas).
3. Para um cliente, soma a similaridade de cada candidato com o que ele já comprou.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity(orders: pd.DataFrame) -> pd.DataFrame:
    """Matriz de similaridade produto x produto (valores entre 0 e 1 para dados não negativos)."""
    # Linhas = clientes, colunas = produtos, valores = quantidade comprada
    matrix = orders.pivot_table(
        index="customer_id",
        columns="product_id",
        values="quantity",
        aggfunc="sum",
        fill_value=0,
    )
    # Cosseno é calculado entre colunas: transpomos para comparar produtos
    sim = cosine_similarity(matrix.T.to_numpy())
    return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)


def recommend(similarity: pd.DataFrame, purchased: list[str], top_n: int = 5) -> list[dict]:
    """Sugere até `top_n` produtos ainda não comprados pelo cliente.

    Lança ValueError se algum produto informado não existir no catálogo.
    """
    unique = list(dict.fromkeys(purchased))  # remove duplicados mantendo a ordem
    unknown = [p for p in unique if p not in similarity.index]
    if unknown:
        raise ValueError(f"Produtos desconhecidos: {unknown}")

    # Pontuação de cada candidato = soma da similaridade com os produtos já comprados
    scores = similarity[unique].sum(axis=1)
    # Não faz sentido recomendar o que o cliente já compra
    scores = scores.drop(index=unique)

    top = scores.sort_values(ascending=False).head(top_n)
    return [{"product_id": pid, "score": round(float(s), 4)} for pid, s in top.items()]
