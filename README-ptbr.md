# Predição de Resultados em League of Legends aos 15 Minutos

Este repositório contém o código-fonte, experimentos e documentação do projeto de comparação de classificadores para predição do time vencedor em partidas de League of Legends (LoL), utilizando apenas dados coletados aos 15 minutos de jogo.

## Sumário
- [Descrição do Projeto](#descrição-do-projeto)
- [Requisitos](#requisitos)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Resultados](#resultados)
- [Artigo Científico](#artigo-científico)
- [Licença](#licença)

---

## Descrição do Projeto
O objetivo deste projeto é comparar algoritmos de aprendizado de máquina para prever, com base em um conjunto restrito de variáveis, qual time vencerá uma partida de LoL. O foco está em cenários de predição precoce, usando apenas informações disponíveis aos 15 minutos de jogo.

Os classificadores avaliados são:
- Árvore de Decisão (DT)
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- Random Forest (RF)
- Ensemble Heterogeneous Boosting (HB)

O código realiza pré-processamento, treinamento, validação cruzada aninhada, otimização de hiperparâmetros e análise estatística dos resultados.

## Requisitos
- Python 3.8+
- Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Como Executar
1. Certifique-se de que o arquivo de dados `data/jogosLoL2021.csv` está presente na pasta do projeto.
2. Execute o script principal:

```bash
python src/tarefa_iv_predicao_15min.py
```

3. (Opcional) Explore e replique os experimentos no Jupyter Notebook:

```bash
jupyter notebook src/Trab1_Tarefa_IV.ipynb
```

## Estrutura do Projeto
```
Primeiro Trabalho - Aprendizado de Máquina/
├── .gitignore                          # Arquivo de exclusões do Git
├── README.md                           # Este arquivo em inglês
├── README-ptbr.md                      # Este arquivo em português
├── requirements.txt                    # Dependências do projeto
├── data/                               # Diretório de dados
│   └── jogosLoL2021.csv                # Base de dados utilizada
├── src/                                # Diretório do código-fonte
│   ├── tarefa_iv_predicao_15min.py     # Script principal dos experimentos
│   └── Trab1_Tarefa_IV.ipynb           # Notebook com análise e visualizações
├── results/                            # Diretório de resultados
│   ├── tabela_resultados.csv           # Resultados das execuções
│   ├── tabela_pvalues.csv              # P-values dos testes estatísticos
│   ├── boxplot_acuracias.png           # Gráfico dos resultados
│   └── matriz_correlacao.png           # Matriz de correlação das features
└── docs/                               # Diretório de documentação
    ├── article-ptbr.pdf                # Artigo científico (Português do Brasil)
    ├── article-en.pdf                  # Artigo científico (Inglês)
    ├── Trab1 IA 2025.md                # Descrição da tarefa
    └── ref/                            # Diretório de referências
        └── *.pdf                       # Artigos de referência
```

## Resultados
Os resultados completos, tabelas e gráficos podem ser encontrados nos arquivos `results/tabela_resultados.csv`, `results/tabela_pvalues.csv` e `results/boxplot_acuracias.png`. O notebook `src/Trab1_Tarefa_IV.ipynb` permite explorar e reproduzir toda a análise.

### Análise de Correlação das Features
A matriz de correlação abaixo mostra as relações entre as features utilizadas aos 15 minutos:

![Matriz de Correlação](results/matriz_correlacao.png)

As features apresentam as seguintes correlações com o resultado da partida:
- **golddiffat15**: 0.5596 (maior correlação)
- **xpdiffat15**: 0.5096
- **killsdiffat15**: 0.4843
- **assistsdiffat15**: 0.4311
- **csdiffat15**: 0.4027 (menor correlação)

## Artigo Científico
O artigo descrevendo a metodologia, experimentos e análise dos resultados está disponível neste repositório:
- [Português (Brasil)](docs/article-ptbr.pdf)
- [Inglês](docs/article-en.pdf)

## Licença
Este projeto é de uso acadêmico. Consulte o arquivo do artigo para detalhes de autoria e citação.
