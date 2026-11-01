# Food Intelligence

Plataforma experimental de Machine Learning para analisar dados sintéticos de uma distribuidora B2B de alimentos. O projeto reúne módulos para previsão de churn, recomendação de produtos e análise de elasticidade-preço, além de uma API baseada em FastAPI.

> **Aviso:** os dados são sintéticos e os modelos são demonstrativos. Resultados não devem ser usados para decisões comerciais reais sem validação, avaliação e revisão de domínio.

## Objetivos

- Gerar um conjunto de dados sintéticos reproduzível para experimentos.
- Preparar atributos para análise de clientes e produtos.
- Explorar modelos de churn, recomendação e elasticidade-preço.
- Expor funcionalidades selecionadas por uma API.
- Manter testes automatizados para reduzir regressões.

## Stack

- Python 3.10+
- FastAPI
- Pydantic
- pytest
- Dependências de análise e Machine Learning declaradas em `requirements.txt` e `pyproject.toml`

## Configuração local

```bash
python -m venv .venv
```

Ative o ambiente:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Testes

```bash
pytest
```

## API

A API está em `src/food_intelligence/api.py`. Depois de instalar o pacote, execute:

```bash
uvicorn food_intelligence.api:app --reload
```

A documentação interativa fica em `http://127.0.0.1:8000/docs`.

Confirme o módulo de importação e a configuração do projeto caso execute a API a partir de outro diretório.

## Organização do código

- `api.py`: camada HTTP.
- `config.py`: configuração do projeto.
- `data_generator.py`: geração de dados sintéticos.
- `features.py`: preparação de atributos.
- `churn.py`: experimento de churn.
- `recommender.py`: lógica de recomendação.
- `elasticity.py`: experimento de elasticidade-preço.
- `train.py`: fluxo de treinamento.
- `tests/`: testes automatizados.

## Reprodutibilidade e qualidade

- Prefira sementes aleatórias configuráveis para geração de dados.
- Separe treino e avaliação para evitar vazamento de dados.
- Compare modelos com métricas adequadas ao problema.
- Documente hipóteses, limitações e versões das dependências.
- Não inclua dados de clientes reais, credenciais ou segredos no Git.

## Próximas melhorias

- Documentar esquemas de entrada/saída dos endpoints.
- Adicionar baseline simples e métricas de avaliação reproduzíveis.
- Criar workflow CI para testes e verificações de estilo.
- Versionar datasets sintéticos e relatórios de experimentos sem incluir artefatos grandes desnecessários.

Projeto de portfólio em desenvolvimento.
