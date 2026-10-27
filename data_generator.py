"""Gerador de dados sintéticos para uma distribuidora B2B de alimentos.

Gera três tabelas:
- products.csv:  catálogo de produtos
- customers.csv: clientes B2B (restaurantes, mercados, padarias, lanchonetes)
- orders.csv:    uma linha por item vendido

Colunas com sufixo "_real" são o *gabarito* da simulação. Elas servem para
conferir se os modelos aprenderam certo, mas NUNCA devem ser usadas como
feature nos modelos (isso seria trapaça).

A simulação embute três padrões que os módulos de ML precisam descobrir:
1. Clientes com pouca frequência de compra tendem a abandonar a base (churn).
2. Clientes compram principalmente de 2 categorias favoritas (recomendação).
3. Quando o preço sobe, a quantidade cai, com sensibilidade diferente por
   produto (elasticidade-preço).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from food_intelligence import config

# Preço-base (R$) por categoria. Cada produto parte daqui com uma variação.
CATEGORY_BASE_PRICE = {
    "laticinios": 12.0,
    "carnes": 35.0,
    "graos": 8.0,
    "bebidas": 6.0,
    "hortifruti": 5.0,
    "congelados": 18.0,
}

SEGMENTS = ["restaurante", "mercado", "padaria", "lanchonete"]
SEGMENT_PROBS = [0.35, 0.25, 0.25, 0.15]
# Fator de volume por segmento: mercados compram bem mais que lanchonetes
SEGMENT_VOLUME = {"restaurante": 1.5, "mercado": 2.0, "padaria": 1.0, "lanchonete": 0.8}
REGIONS = ["norte", "nordeste", "centro-oeste", "sudeste", "sul"]


def generate_dataset(
    n_customers: int = 800,
    n_products: int = 40,
    seed: int = config.RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Gera as três tabelas sintéticas de forma reproduzível (mesma seed = mesmos dados)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(config.SIMULATION_START)
    horizon = config.SIMULATION_DAYS

    # ---------- 1. Produtos ----------
    categories = rng.choice(list(CATEGORY_BASE_PRICE), size=n_products)
    base_prices = np.array([CATEGORY_BASE_PRICE[c] for c in categories]) * rng.uniform(0.7, 1.5, n_products)
    products = pd.DataFrame(
        {
            "product_id": [f"P{i:03d}" for i in range(n_products)],
            "category": categories,
            "base_price": base_prices.round(2),
            # Elasticidade verdadeira: sempre negativa (preço sobe -> demanda cai).
            # Valores entre -0.3 (inelástico, ex.: arroz) e -2.5 (muito elástico).
            "elasticidade_real": rng.uniform(-2.5, -0.3, n_products).round(3),
            # Volume típico vendido por item de pedido
            "demanda_base_real": rng.integers(5, 40, n_products),
        }
    )

    # ---------- 2. Clientes ----------
    segments = rng.choice(SEGMENTS, size=n_customers, p=SEGMENT_PROBS)
    regions = rng.choice(REGIONS, size=n_customers)
    signup_day = rng.integers(0, 365, n_customers)  # entram ao longo do 1º ano
    # Ritmo de compra latente (pedidos por mês): propensão de cada cliente
    orders_per_month = rng.gamma(shape=2.0, scale=1.2, size=n_customers) + 0.2
    # Quem compra pouco tem mais chance de sair da base
    churn_prob = 0.7 / (1 + np.exp(1.2 * (orders_per_month - 1.5)))
    will_churn = rng.random(n_customers) < churn_prob
    # Dia de saída (se churnar): entre 4 meses após a entrada e o fim da simulação
    churn_day = rng.integers(low=signup_day + 120, high=horizon)
    stop_day = np.where(will_churn, churn_day, horizon)

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n_customers)],
            "segment": segments,
            "region": regions,
            "signup_date": start + pd.to_timedelta(signup_day, unit="D"),
        }
    )

    # ---------- 3. Pedidos ----------
    cat_names = sorted(CATEGORY_BASE_PRICE)
    prod_ids = products["product_id"].to_numpy()
    prod_base_price = products["base_price"].to_numpy()
    prod_elasticity = products["elasticidade_real"].to_numpy()
    prod_volume = products["demanda_base_real"].to_numpy()
    prod_category = products["category"].to_numpy()

    rows: list[tuple] = []
    order_counter = 0
    for i in range(n_customers):
        customer_id = customers.at[i, "customer_id"]
        segment = segments[i]
        rate_per_day = orders_per_month[i] / 30

        # Cada cliente tem 2 categorias favoritas: é isso que cria padrões de recomendação
        fav_cats = rng.choice(cat_names, size=2, replace=False)
        fav_idx = np.flatnonzero(np.isin(prod_category, fav_cats))

        t = float(signup_day[i])
        while True:
            # Intervalo entre pedidos segue uma distribuição exponencial
            t += rng.exponential(1 / rate_per_day)
            if t >= stop_day[i]:
                break

            order_id = f"O{order_counter:07d}"
            order_counter += 1
            order_date = start + pd.Timedelta(days=int(t))

            for _ in range(int(rng.integers(1, 5))):  # 1 a 4 itens por pedido
                if rng.random() < 0.7:
                    p = int(rng.choice(fav_idx))  # 70%: produto favorito
                else:
                    p = int(rng.integers(n_products))  # 30%: exploração

                # Preço varia ±20% ao longo do tempo (promoções, reajustes)
                price = prod_base_price[p] * rng.uniform(0.8, 1.2)
                # Quantidade: volume típico * efeito do preço * ruído aleatório
                expected = SEGMENT_VOLUME[segment] * prod_volume[p]
                expected *= (price / prod_base_price[p]) ** prod_elasticity[p]
                quantity = max(1, int(round(expected * np.exp(rng.normal(0, 0.25)))))

                rows.append((order_id, customer_id, prod_ids[p], order_date, quantity, round(price, 2)))

    orders = pd.DataFrame(
        rows,
        columns=["order_id", "customer_id", "product_id", "order_date", "quantity", "unit_price"],
    )

    return {"products": products, "customers": customers, "orders": orders}


def save_tables(tables: dict[str, pd.DataFrame], data_dir: Path) -> None:
    """Salva as tabelas como CSV na pasta indicada."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(data_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Lê os CSVs gerados, convertendo as colunas de data."""
    return {
        "products": pd.read_csv(data_dir / "products.csv"),
        "customers": pd.read_csv(data_dir / "customers.csv", parse_dates=["signup_date"]),
        "orders": pd.read_csv(data_dir / "orders.csv", parse_dates=["order_date"]),
    }
