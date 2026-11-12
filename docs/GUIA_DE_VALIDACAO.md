# Verificação e evolução — Food Intelligence

## Escopo confirmado

Protótipo de ML para dados B2B sintéticos com API FastAPI, modelos e avaliação.

## Localização do código

`src/food_intelligence/api.py`, `train.py`, `churn.py`, `recommender.py`, `elasticity.py` e testes Python.

## Verificar localmente

Instalar dependências de `requirements.txt`, conferir comandos oficiais no README, executar `python -m pytest -q`.

Use dados fictícios e registre no PR o comando, ambiente e resultado observado. Para projetos sem runtime instalado, marque a verificação como pendente em vez de afirmar que passou.

## Casos de regressão a cobrir

Validar separação treino/teste, dados ausentes, vazamento de atributos, entradas adversariais da API e métricas por corte temporal.

## Limitações e próximo incremento

Modelos e dados são demonstrativos. Próximo incremento: registrar seed, esquema de features, métricas e versionamento de artefatos antes de interpretar previsões.

## Critérios de revisão

- [ ] O comportamento alterado está descrito com exemplo reproduzível.
- [ ] Há teste ou procedimento de validação compatível com o código real.
- [ ] Nenhuma chave, dado pessoal ou arquivo gerado foi incluído sem necessidade.
- [ ] A documentação distingue o implementado do planejado.
