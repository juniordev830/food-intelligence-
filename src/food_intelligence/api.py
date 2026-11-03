"""API de predições — FastAPI.

Rodar localmente (ou no Codespaces):
    uvicorn food_intelligence.api:app --reload --port 8000

Documentação interativa (Swagger): http://localhost:8000/docs

Os modelos são carregados da pasta `artifacts/` na primeira vez que são usados.
Por isso, você NÃO precisa reiniciar o servidor depois de treinar de novo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from food_intelligence import config
from food_intelligence.churn import predict_churn_proba, risk_level
from food_intelligence.elasticity import simulate_price_change
from food_intelligence.features import FEATURE_COLUMNS
from food_intelligence.recommender import recommend

TRAIN_HINT = "Modelo ainda não treinado. Rode: python -m food_intelligence.train"


# ---------- Schemas: definem e validam o formato de entrada e saída ----------

class ChurnRequest(BaseModel):
    """Features de um cliente (mesmas colunas de FEATURE_COLUMNS)."""

    recency_days: int = Field(ge=0, description="Dias desde o último pedido")
    frequency_90d: int = Field(ge=0, description="Pedidos nos últimos 90 dias")
    monetary_90d: float = Field(ge=0, allow_inf_nan=False, description="Faturamento nos últimos 90 dias (R$)")
    avg_order_value: float = Field(ge=0, allow_inf_nan=False, description="Ticket médio por pedido (R$)")
    n_categories: int = Field(ge=0, description="Categorias diferentes compradas")
    tenure_days: int = Field(ge=0, description="Dias desde o primeiro pedido")


class ChurnResponse(BaseModel):
    churn_probability: float
    risk_level: str
    will_churn: bool  # True se a probabilidade for >= 50%


class RecommendationRequest(BaseModel):
    purchased_product_ids: list[str] = Field(min_length=1, max_length=500, description="Produtos que o cliente já comprou (máximo de 500 itens)")
    top_n: int = Field(default=5, ge=1, le=20)

    @field_validator("purchased_product_ids")
    @classmethod
    def normalize_purchased_product_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("IDs de produto não podem ser vazios")
        if any(len(value) > 100 for value in normalized):
            raise ValueError("IDs de produto devem ter no máximo 100 caracteres")
        return normalized


class RecommendationItem(BaseModel):
    product_id: str
    score: float


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]


class ElasticityRequest(BaseModel):
    product_id: str = Field(min_length=1, max_length=100)
    current_price: float = Field(gt=0, allow_inf_nan=False, description="Preço atual (R$)")
    new_price: float = Field(gt=0, allow_inf_nan=False, description="Preço proposto (R$)")
    current_quantity: float = Field(gt=0, allow_inf_nan=False, description="Quantidade vendida hoje")

    @field_validator("product_id")
    @classmethod
    def normalize_product_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ID de produto não pode ser vazio")
        return normalized


class ElasticityResponse(BaseModel):
    product_id: str
    elasticity: float
    classification: str
    predicted_quantity: float
    revenue_change_pct: float


# ---------- Carregamento sob demanda dos artefatos ----------

class ModelStore:
    """Guarda os artefatos em memória, lendo do disco só quando necessário."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._cache: dict[str, tuple[int, int, Any]] = {}

    def get(self, name: str) -> Any:
        path = self.directory / f"{name}.joblib"
        try:
            stat = path.stat()
        except FileNotFoundError:
            raise FileNotFoundError(path) from None

        cached = self._cache.get(name)
        signature = (stat.st_mtime_ns, stat.st_size)
        if cached is None or cached[:2] != signature:
            self._cache[name] = (*signature, joblib.load(path))
        return self._cache[name][2]


def _load_or_503(store: ModelStore, name: str) -> Any:
    """Se o modelo não existe, responde 503 (serviço indisponível) com instrução clara."""
    try:
        return store.get(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=TRAIN_HINT) from exc


# ---------- Fábrica da aplicação ----------

def create_app(artifacts_dir: Path = config.ARTIFACTS_DIR) -> FastAPI:
    """Cria a aplicação. Separar em função facilita os testes (podemos apontar outra pasta)."""
    app = FastAPI(
        title="Food Intelligence API",
        version="0.1.0",
        description="Predições de churn, recomendação de produtos e elasticidade-preço.",
    )
    store = ModelStore(artifacts_dir)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Verifica se a API está no ar."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        """Indica se os artefatos necessários para cada predição estão disponíveis."""
        model_names = ("churn_model", "recommender", "elasticity")
        models = {
            name: (artifacts_dir / f"{name}.joblib").is_file()
            for name in model_names
        }
        return {
            "status": "ready" if all(models.values()) else "degraded",
            "models": models,
            "training_hint": TRAIN_HINT if not all(models.values()) else None,
        }

    @app.post("/churn/predict", response_model=ChurnResponse)
    def predict_churn(body: ChurnRequest) -> ChurnResponse:
        """Probabilidade de um cliente parar de comprar nos próximos 60 dias."""
        model = _load_or_503(store, "churn_model")
        # Monta uma linha com as colunas na mesma ordem usada no treino
        row = pd.DataFrame([body.model_dump()])[FEATURE_COLUMNS]
        probability = float(predict_churn_proba(model, row)[0])
        return ChurnResponse(
            churn_probability=round(probability, 4),
            risk_level=risk_level(probability),
            will_churn=probability >= 0.5,
        )

    @app.post("/recommendations", response_model=RecommendationResponse)
    def get_recommendations(body: RecommendationRequest) -> RecommendationResponse:
        """Produtos parecidos com os que o cliente já compra."""
        similarity = _load_or_503(store, "recommender")
        try:
            items = recommend(similarity, body.purchased_product_ids, body.top_n)
        except ValueError as exc:  # produto que não existe no catálogo
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RecommendationResponse(recommendations=items)

    @app.post("/elasticity/simulate", response_model=ElasticityResponse)
    def simulate_elasticity(body: ElasticityRequest) -> ElasticityResponse:
        """Prevê quantidade e receita se o preço do produto mudar."""
        table = _load_or_503(store, "elasticity")
        if body.product_id not in table.index:
            raise HTTPException(status_code=404, detail=f"Sem elasticidade estimada para {body.product_id}")

        elasticity = float(table.loc[body.product_id, "elasticity"])
        try:
            result = simulate_price_change(
                elasticity, body.current_price, body.new_price, body.current_quantity
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ElasticityResponse(product_id=body.product_id, elasticity=round(elasticity, 4), **result)

    return app


# Objeto usado pelo uvicorn: `uvicorn food_intelligence.api:app`
app = create_app()
