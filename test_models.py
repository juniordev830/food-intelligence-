"""Testes dos módulos de ML e do pipeline de treinamento."""

import numpy as np
import pandas as pd
import pytest

from food_intelligence.churn import risk_level, train_churn_model
from food_intelligence.data_generator import generate_dataset
from food_intelligence.elasticity import estimate_elasticities, simulate_price_change
from food_intelligence.features import FEATURE_COLUMNS, build_churn_labels, build_customer_features
from food_intelligence.recommender import build_similarity, recommend
from food_intelligence.train import reference_date, run_pipeline


@pytest.fixture(scope="module")
def data():
    return generate_dataset(n_customers=200, seed=0)


# ---------- Features e rótulo ----------

def test_features_tem_ordem_fixa_e_uma_linha_por_cliente(data):
    X = build_customer_features(data["orders"], data["products"], reference_date())
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.index.is_unique
    assert X.notna().all().all()
    assert (X["recency_days"] >= 0).all()


def test_features_nao_enxergam_o_futuro(data):
    """Se mudarmos pedidos depois da data de corte, as features não podem mudar."""
    ref = reference_date()
    altered = data["orders"].copy()
    altered.loc[altered["order_date"] >= ref, "quantity"] *= 100

    X_original = build_customer_features(data["orders"], data["products"], ref)
    X_altered = build_customer_features(altered, data["products"], ref)
    pd.testing.assert_frame_equal(X_original, X_altered)


def test_rotulo_de_churn_bate_com_compras_na_janela_futura(data):
    ref = reference_date()
    X = build_customer_features(data["orders"], data["products"], ref)
    y = build_churn_labels(data["orders"], X.index, ref)

    assert set(y.unique()) <= {0, 1}
    end = ref + pd.Timedelta(days=60)
    buyers = set(data["orders"].loc[
        (data["orders"]["order_date"] >= ref) & (data["orders"]["order_date"] < end), "customer_id"
    ])
    # rótulo 0 = comprou na janela futura; rótulo 1 = não comprou
    assert all((cid in buyers) == (label == 0) for cid, label in y.items())


# ---------- Churn ----------

def test_modelo_de_churn_aprende_algum_sinal(data):
    ref = reference_date()
    X = build_customer_features(data["orders"], data["products"], ref)
    y = build_churn_labels(data["orders"], X.index, ref)

    _, metrics = train_churn_model(X, y, seed=0)
    assert 0.5 < metrics["roc_auc"] <= 1.0
    assert metrics["n_train"] > metrics["n_test"]


def test_faixas_de_risco():
    assert risk_level(0.10) == "baixo"
    assert risk_level(0.45) == "médio"
    assert risk_level(0.90) == "alto"


# ---------- Recomendação ----------

def test_recomendacao_nao_repete_compras_e_respeita_top_n(data):
    sim = build_similarity(data["orders"])
    assert sim.shape[0] == sim.shape[1] == len(data["products"])

    purchased = [sim.index[0], sim.index[1]]
    recs = recommend(sim, purchased, top_n=5)
    ids = [r["product_id"] for r in recs]

    assert len(ids) == 5
    assert not set(ids) & set(purchased)
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_recomendacao_rejeita_produto_inexistente(data):
    sim = build_similarity(data["orders"])
    with pytest.raises(ValueError):
        recommend(sim, ["NAO_EXISTE"])


# ---------- Elasticidade ----------

def test_elasticidade_estimada_acompanha_o_gabarito(data):
    table = estimate_elasticities(data["orders"])
    truth = data["products"].set_index("product_id")["elasticidade_real"]
    joined = table.join(truth, how="inner")

    assert (table["elasticity"] < 0).mean() > 0.9
    assert np.corrcoef(joined["elasticity"], joined["elasticidade_real"])[0, 1] > 0.8


def test_aumento_de_preco_reduz_a_quantidade_com_elasticidade_negativa():
    result = simulate_price_change(-1.5, current_price=10, new_price=12, current_quantity=100)
    assert result["predicted_quantity"] < 100
    assert result["revenue_change_pct"] < 0
    assert result["classification"] == "elástica"


def test_demanda_inelastica_e_classificada_corretamente():
    result = simulate_price_change(-0.5, current_price=10, new_price=11, current_quantity=100)
    assert result["classification"] == "inelástica"


def test_sem_mudanca_de_preco_a_quantidade_nao_muda():
    result = simulate_price_change(-1.5, current_price=10, new_price=10, current_quantity=100)
    assert result["predicted_quantity"] == 100
    assert result["revenue_change_pct"] == 0


# ---------- Pipeline ----------

def test_pipeline_gera_todos_os_artefatos(tmp_path):
    metrics = run_pipeline(
        n_customers=100, seed=1, data_dir=tmp_path / "raw", artifacts_dir=tmp_path / "art"
    )
    for name in ["churn_model.joblib", "recommender.joblib", "elasticity.joblib", "metrics.json"]:
        assert (tmp_path / "art" / name).exists(), f"faltou {name}"
    assert "roc_auc" in metrics["churn"]
