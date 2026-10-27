"""Fixtures compartilhadas pelos testes.

Usamos poucos clientes para os testes rodarem em segundos. Os arquivos ficam
em pastas temporárias, então não tocamos em `data/` nem em `artifacts/` reais.
"""

import pytest

from food_intelligence.train import run_pipeline


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    """Treina o pipeline inteiro uma única vez e devolve a pasta de artefatos."""
    base = tmp_path_factory.mktemp("food_intelligence")
    run_pipeline(
        n_customers=150,
        seed=0,
        data_dir=base / "raw",
        artifacts_dir=base / "artifacts",
    )
    return base / "artifacts"
