"""Configurações centrais do projeto.

Os caminhos podem ser sobrescritos por variáveis de ambiente, o que facilita
rodar testes em pastas temporárias sem mexer nos arquivos reais.
"""

import os
from pathlib import Path

# Raiz do repositório: src/food_intelligence/config.py -> sobe 3 níveis
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("FI_DATA_DIR", ROOT_DIR / "data" / "raw"))
ARTIFACTS_DIR = Path(os.getenv("FI_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))

# Semente para tornar os dados e os treinos reproduzíveis
RANDOM_SEED = 42

# Início da simulação e tamanho do histórico gerado
SIMULATION_START = "2023-01-01"
SIMULATION_DAYS = 730  # ~2 anos de pedidos

# Janela de observação: quanto histórico usamos para montar as features do cliente
OBSERVATION_DAYS = 90

# Janela de previsão: se o cliente não comprar nesses dias, consideramos churn
CHURN_WINDOW_DAYS = 60
