# Food Intelligence

Plataforma de Machine Learning que simula um problema real de uma **distribuidora B2B de alimentos**: clientes como restaurantes, mercados, padarias e lanchonetes compram em volume, e a empresa precisa saber quem está prestes a sair, o que sugerir e como o preço afeta as vendas.

O projeto tem três módulos, um pipeline de treinamento, uma API REST e testes automatizados. Os dados são **sintéticos**, gerados pelo próprio projeto.

| Módulo | Pergunta de negócio | Modelo | Métrica |
|---|---|---|---|
| **Churn** | Quais clientes têm alta chance de parar de comprar nos próximos 60 dias? | RandomForest (scikit-learn) | ROC AUC |
| **Recomendação** | Que produtos sugerir para um cliente? | Filtragem colaborativa item-a-item (similaridade de cosseno) | Score de similaridade |
| **Elasticidade** | Se o preço do produto X subir 10%, o que acontece com a quantidade e a receita? | Regressão log-log, uma por produto | Elasticidade e R² |

A decisão de arquitetura (por que FastAPI, scikit-learn e joblib) está em [`docs/adr/0001-stack-treino-e-api.md`](docs/adr/0001-stack-treino-e-api.md).

## Estrutura do projeto

```
food-intelligence/
├── .devcontainer/devcontainer.json   # configuração do GitHub Codespaces
├── docs/adr/0001-stack-treino-e-api.md
├── src/food_intelligence/
│   ├── config.py           # caminhos e parâmetros globais
│   ├── data_generator.py   # gera os dados sintéticos
│   ├── features.py         # features por cliente e rótulo de churn
│   ├── churn.py            # módulo 1: churn
│   ├── recommender.py      # módulo 2: recomendação
│   ├── elasticity.py       # módulo 3: elasticidade-preço
│   ├── train.py            # pipeline de treinamento
│   └── api.py              # API FastAPI
├── tests/                  # testes com pytest
├── requirements.txt
└── pyproject.toml
```

## Dados sintéticos

O gerador cria três tabelas em `data/raw/`:

- **products.csv**: catálogo de 40 produtos em 6 categorias.
- **customers.csv**: 800 clientes, com segmento, região e data de cadastro.
- **orders.csv**: cerca de 90 mil itens de pedido ao longo de 2 anos.

A simulação embute padrões que os modelos precisam descobrir: clientes que compram pouco têm mais chance de sair; cada cliente tem duas categorias favoritas; e a quantidade vendida cai quando o preço sobe, com sensibilidade diferente em cada produto.

As colunas com sufixo `_real` (como `elasticidade_real`) são o **gabarito** da simulação. Servem para conferir se os modelos aprenderam certo. Elas nunca entram como feature.

## Passo a passo no GitHub Codespaces

**1. Crie o repositório no GitHub.** Clique em **New**, dê um nome (por exemplo `food-intelligence`) e deixe-o **vazio**, sem README e sem .gitignore.

**2. Envie o código.** Na pasta descompactada do projeto, rode:

```bash
git init -b main
git add .
git commit -m "feat: primeira versão do Food Intelligence"
git remote add origin https://github.com/SEU-USUARIO/food-intelligence.git
git push -u origin main
```

**3. Abra um Codespace.** Na página do repositório, clique em **Code → Codespaces → Create codespace on main**. Na primeira vez, o ambiente instala as dependências sozinho. Isso leva de 1 a 3 minutos.

**4. Treine os modelos.** No terminal do Codespace, rode:

```bash
python -m food_intelligence.train
```

O comando gera os dados, treina os três módulos e salva os resultados em `artifacts/`. Com a configuração padrão (800 clientes, semente 42), você deve ver `roc_auc` próximo de **0,86**. Os valores são de referência: pequenas diferenças de ambiente podem alterá-los levemente.

**5. Rode os testes.**

```bash
pytest -q
```

Você deve ver **24 testes passando**.

**6. Suba a API.**

```bash
uvicorn food_intelligence.api:app --reload --port 8000
```

**7. Abra a documentação interativa.** Vá à aba **PORTS**, clique no ícone de globo ao lado da porta **8000** e acrescente `/docs` ao endereço. Você verá o Swagger, onde é possível testar cada endpoint direto no navegador.

## Exemplos de chamadas

Com a API no ar, você pode chamar os endpoints pelo terminal. Para ver o formato completo, consulte `/docs`.

```bash
# Churn: probabilidade de um cliente parar de comprar
curl -X POST localhost:8000/churn/predict -H "Content-Type: application/json" \
  -d '{"recency_days": 95, "frequency_90d": 1, "monetary_90d": 200, "avg_order_value": 200, "n_categories": 1, "tenure_days": 300}'

# Recomendação: produtos parecidos com os que o cliente já compra
curl -X POST localhost:8000/recommendations -H "Content-Type: application/json" \
  -d '{"purchased_product_ids": ["P000", "P001"], "top_n": 3}'

# Elasticidade: efeito de subir o preço de R$ 10 para R$ 12
curl -X POST localhost:8000/elasticity/simulate -H "Content-Type: application/json" \
  -d '{"product_id": "P001", "current_price": 10, "new_price": 12, "current_quantity": 100}'
```

A API só usa os modelos treinados. Se você tentar uma chamada antes de rodar o treino, recebe `503` com a instrução para treinar. Como os modelos são carregados sob demanda, você não precisa reiniciar o servidor depois de treinar de novo.

## Rodar fora do Codespaces

Com Python 3.11 ou superior:

```bash
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m food_intelligence.train
pytest -q
uvicorn food_intelligence.api:app --reload --port 8000
```

## Testes

Os testes ficam em `tests/` e cobrem:

- **Dados:** colunas, ausência de nulos, reprodutibilidade pela semente e coerência das datas.
- **Features sem vazamento:** um teste altera pedidos *depois* da data de corte e verifica que as features não mudam.
- **Modelos:** o churn aprende algum sinal, a recomendação não repete compras e a elasticidade estimada acompanha o gabarito.
- **API:** respostas de sucesso, validação de entrada (422), produto inexistente (404) e ausência de modelos (503).

## Limitações e cuidados

- **Divisão aleatória:** o teste de churn usa uma divisão aleatória. Em produção, o correto é uma divisão no tempo, para treinar no passado e testar no futuro.
- **Elasticidade é correlação:** se o preço subiu justamente em épocas de alta demanda, a estimativa fica enviesada. Para uma decisão real de preço, o ideal é um teste A/B.
- **Dados sintéticos:** os números servem para estudar a técnica, não para tirar conclusões sobre um mercado real.
- **Sem autenticação nem banco de dados:** a API é aberta e os artefatos ficam em arquivos locais. Isso é adequado para estudo, não para produção.

## Próximos passos sugeridos

- Trocar a divisão aleatória por uma divisão temporal e validação cruzada.
- Registrar experimentos e versões de modelo com MLflow.
- Criar um `Dockerfile` para deploy.
- Montar um painel com Streamlit para o time comercial.
