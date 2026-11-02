"""Testes da API usando o TestClient (sem precisar subir o servidor)."""

import os

import joblib
import pytest
from fastapi.testclient import TestClient

from food_intelligence.api import ModelStore, create_app

CHURN_PAYLOAD = {
    "recency_days": 10,
    "frequency_90d": 4,
    "monetary_90d": 1500.0,
    "avg_order_value": 375.0,
    "n_categories": 3,
    "tenure_days": 400,
}


@pytest.fixture(scope="module")
def client(artifacts_dir):
    return TestClient(create_app(artifacts_dir))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_churn_retorna_probabilidade_e_risco(client):
    response = client.post("/churn/predict", json=CHURN_PAYLOAD)
    body = response.json()

    assert response.status_code == 200
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"baixo", "médio", "alto"}
    assert body["will_churn"] == (body["churn_probability"] >= 0.5)


def test_churn_rejeita_valores_invalidos(client):
    response = client.post("/churn/predict", json={**CHURN_PAYLOAD, "recency_days": -1})
    assert response.status_code == 422  # erro de validação do Pydantic


def test_recomendacoes_retornam_top_n(client):
    response = client.post(
        "/recommendations", json={"purchased_product_ids": ["P000"], "top_n": 3}
    )
    body = response.json()

    assert response.status_code == 200
    assert len(body["recommendations"]) == 3
    assert "P000" not in [item["product_id"] for item in body["recommendations"]]


def test_recomendacao_com_produto_desconhecido_retorna_404(client):
    response = client.post("/recommendations", json={"purchased_product_ids": ["XXX"]})
    assert response.status_code == 404


def test_elasticidade_aumento_de_preco_reduz_quantidade(client):
    response = client.post(
        "/elasticity/simulate",
        json={"product_id": "P000", "current_price": 10.0, "new_price": 11.0, "current_quantity": 100.0},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["elasticity"] < 0
    assert body["predicted_quantity"] < 100



def test_recomendacoes_normalizam_espacos_em_ids(client):
    response = client.post(
        "/recommendations", json={"purchased_product_ids": [" P000 "], "top_n": 3}
    )
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 3
    assert "P000" not in [item["product_id"] for item in response.json()["recommendations"]]


def test_recomendacoes_rejeitam_id_vazio_ou_so_com_espacos(client):
    response = client.post(
        "/recommendations", json={"purchased_product_ids": ["   "]}
    )
    assert response.status_code == 422


def test_elasticidade_normaliza_id_do_produto(client):
    response = client.post(
        "/elasticity/simulate",
        json={
            "product_id": " P000 ",
            "current_price": 10.0,
            "new_price": 11.0,
            "current_quantity": 100.0,
        },
    )
    assert response.status_code == 200
    assert response.json()["product_id"] == "P000"


def test_elasticidade_rejeita_id_vazio_ou_so_com_espacos(client):
    response = client.post(
        "/elasticity/simulate",
        json={
            "product_id": "   ",
            "current_price": 10.0,
            "new_price": 11.0,
            "current_quantity": 100.0,
        },
    )
    assert response.status_code == 422


def test_sem_modelos_treinados_a_api_responde_503(tmp_path):
    empty_client = TestClient(create_app(tmp_path))
    response = empty_client.post("/churn/predict", json=CHURN_PAYLOAD)
    assert response.status_code == 503
    assert "food_intelligence.train" in response.json()["detail"]



def test_model_store_reloads_artifact_when_file_changes(tmp_path):
    artifact = tmp_path / "sample.joblib"
    joblib.dump({"version": 1}, artifact)
    store = ModelStore(tmp_path)

    assert store.get("sample") == {"version": 1}
    previous_stat = artifact.stat()

    joblib.dump({"version": 2}, artifact)
    os.utime(
        artifact,
        ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000),
    )

    assert store.get("sample") == {"version": 2}
