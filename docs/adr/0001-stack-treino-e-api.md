# ADR-0001: Stack para treino e serviço de predições

**Status:** Accepted
**Date:** 2026-10-27
**Deciders:** Autor do projeto

## Context

Precisamos de uma plataforma que:

- treine três modelos de Machine Learning com dados tabulares (churn, recomendação e elasticidade);
- exponha as predições por HTTP de forma simples;
- rode no GitHub Codespaces, sem serviços pagos nem infraestrutura externa;
- seja reprodutível e fácil de ler, já que é também material de estudo.

Restrições: o treino completo precisa levar poucos minutos, e o código deve funcionar com Python 3.11.

## Decision

Usar **pandas e numpy** para manipulação de dados, **scikit-learn** para os modelos (com `Pipeline` para empacotar pré-processamento e modelo), **joblib** para salvar os artefatos em disco e **FastAPI** para expor a API. As versões ficam fixadas em `requirements.txt`.

## Options Considered

### Option A: FastAPI + scikit-learn + joblib (escolhida)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Baixa |
| Cost | Zero: roda inteiro no Codespaces |
| Scalability | Adequada para protótipo; escala com containers se necessário |
| Team familiarity | Alta em Python |

**Pros:** validação de entrada com Pydantic (erro 422 automático com mensagem clara); documentação Swagger gerada em `/docs`, útil para testar no navegador; `TestClient` permite testar a API sem subir servidor.
**Cons:** mais um conceito para quem vem do Flask (tipos e schemas); acoplamento às versões do Pydantic.

### Option B: Flask + scikit-learn + joblib

| Dimension | Assessment |
|-----------|------------|
| Complexity | Baixa |
| Cost | Zero |
| Scalability | Menor sem extensões adicionais |
| Team familiarity | Alta |

**Pros:** mais simples e muito conhecido; menos dependências.
**Cons:** validação manual de cada campo; documentação da API precisa ser escrita à mão; sem schemas tipados.

### Option C: FastAPI + MLflow (registro e serving de modelos)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Alta |
| Cost | Exige um servidor de tracking adicional |
| Scalability | Alta |
| Team familiarity | Baixa |

**Pros:** rastreio de experimentos, versionamento de modelos e interface visual.
**Cons:** infraestrutura extra e complexidade excessiva para o escopo atual.

## Trade-off Analysis

**A vs. B:** a diferença principal é a validação e a documentação automáticas do FastAPI. O custo é uma dependência a mais. Escolhemos A porque o Swagger facilita testar a API direto no Codespaces.

**A vs. C:** o MLflow resolve um problema que ainda não existe aqui (muitos experimentos e versões em paralelo). Por isso fica como evolução futura.

**Modelos escolhidos dentro dessa stack:**

- *Churn com RandomForest:* captura relações não lineares sem engenharia manual de variáveis. Uma regressão logística seria o baseline interpretável; como o modelo está dentro de um `Pipeline`, trocar é uma linha.
- *Recomendação item-a-item com cosseno:* a explicação é direta ("clientes que compram X também compram Y"), não exige treino pesado e cabe em memória com algumas centenas de produtos. Fatoração de matrizes (NMF/ALS) fica como alternativa para catálogos maiores.
- *Elasticidade com regressão log-log:* interpretável e rápida. A limitação é que mede correlação em dados observacionais, não causalidade.

## Consequences

**Fica mais fácil:** rodar, ler, testar e trocar modelos dentro do `Pipeline`.

**Fica mais difícil:**

- Escalar para milhões de clientes: a matriz de similaridade é densa e os artefatos ficam em disco local.
- Versionar modelos: hoje é manual (arquivos `.joblib` e `metrics.json`).
- Segurança dos artefatos: `joblib` só deve carregar arquivos de fontes confiáveis, porque pode executar código ao ser lido.

**Revisitar quando:**

- houver mais de um modelo ou mais de uma pessoa treinando: adotar MLflow (Opção C);
- o catálogo passar de cerca de 10 mil produtos: trocar a matriz densa por uma esparsa ou por ALS;
- for para produção: trocar o split aleatório por um split temporal.

## Action Items

1. [x] Estrutura do projeto e pipeline de treino
2. [x] Testes automatizados com pytest
3. [x] Configuração do Codespaces
4. [ ] Split temporal e validação cruzada
5. [ ] Dockerfile para deploy
6. [ ] Avaliar MLflow para registro de experimentos
