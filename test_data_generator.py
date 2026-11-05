"""Testes do gerador de dados sintéticos."""

import pandas as pd
import pytest

from food_intelligence.data_generator import generate_dataset


def _small(seed: int = 1) -> dict:
    return generate_dataset(n_customers=30, n_products=10, seed=seed)


def test_tabelas_tem_as_colunas_esperadas():
    tables = _small()
    assert set(tables["orders"].columns) == {
        "order_id", "customer_id", "product_id", "order_date", "quantity", "unit_price",
    }
    assert set(tables["customers"].columns) == {"customer_id", "segment", "region", "signup_date"}
    assert len(tables["products"]) == 10
    assert len(tables["customers"]) == 30


def test_dados_sem_nulos_e_com_valores_validos():
    orders = _small()["orders"]
    assert orders.isna().sum().sum() == 0
    assert (orders["quantity"] >= 1).all()
    assert (orders["unit_price"] > 0).all()


def test_mesma_seed_gera_exatamente_os_mesmos_dados():
    a = _small(seed=3)["orders"]
    b = _small(seed=3)["orders"]
    pd.testing.assert_frame_equal(a, b)


def test_seeds_diferentes_geram_dados_diferentes():
    a = _small(seed=3)["orders"]
    c = _small(seed=4)["orders"]
    assert not a.equals(c)


def test_cliente_so_compra_depois_do_cadastro():
    tables = _small(seed=2)
    merged = tables["orders"].merge(tables["customers"], on="customer_id")
    assert (merged["order_date"] >= merged["signup_date"]).all()


@pytest.mark.parametrize("n_products", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 5])
def test_tiny_catalog_never_selects_missing_favorite_categories(n_products, seed):
    tables = generate_dataset(n_customers=20, n_products=n_products, seed=seed)
    catalog = set(tables["products"]["product_id"])
    assert len(catalog) == n_products
    assert set(tables["orders"]["product_id"]).issubset(catalog)
    assert tables["orders"]["product_id"].notna().all()


@pytest.mark.parametrize("field,value", [
    ("n_products", 0), ("n_products", -1), ("n_products", True), ("n_products", 1.5),
    ("n_customers", 0), ("n_customers", -2), ("n_customers", False), ("n_customers", 3.5),
])
def test_generator_rejects_invalid_catalog_or_customer_counts(field, value):
    options = {"n_customers": 10, "n_products": 4, "seed": 1}
    options[field] = value
    with pytest.raises(ValueError, match="inteiro positivo"):
        generate_dataset(**options)
