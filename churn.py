"""Módulo 1 — Previsão de churn: qual cliente vai parar de comprar?

Modelo: RandomForest dentro de um Pipeline do scikit-learn.
O Pipeline junta o pré-processamento (StandardScaler) e o modelo em um único
objeto. Isso garante que o mesmo tratamento seja aplicado no treino e na API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(seed: int) -> Pipeline:
    """Cria o pipeline não treinado.

    A escala não é obrigatória para árvores, mas deixa o pipeline pronto
    para trocar o modelo por algo linear (ex.: regressão logística) sem reescrever nada.
    """
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,  # evita árvores decorando casos isolados
            random_state=seed,
            n_jobs=1,  # determinístico e simples de rodar no Codespaces
        ),
    )


def train_churn_model(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[Pipeline, dict]:
    """Treina o modelo e devolve (modelo, métricas).

    Usamos 25% dos clientes como teste. `stratify` mantém a mesma proporção de
    churn nos dois conjuntos, o que deixa a avaliação mais justa.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    model = build_model(seed)
    model.fit(X_train, y_train)

    # Probabilidade de churn (classe 1) nos dados que o modelo nunca viu
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        # ROC AUC: 0.5 = chute, 1.0 = separa perfeitamente quem churna de quem não churna
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
    }
    return model, metrics


def predict_churn_proba(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Retorna a probabilidade de churn de cada linha de `features`."""
    return model.predict_proba(features)[:, 1]


def risk_level(probability: float) -> str:
    """Traduz a probabilidade numérica em uma faixa de risco para o time comercial."""
    if probability < 0.3:
        return "baixo"
    if probability < 0.6:
        return "médio"
    return "alto"
