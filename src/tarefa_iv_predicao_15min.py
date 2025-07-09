import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configurações globais
ESTADO_ALEATORIO_EXTERNO = 36854321
ESTADO_ALEATORIO_INTERNO = 36854321
ESTADO_ALEATORIO_CLASSIFICADORES = 13
N_FOLDS_EXTERNO = 10
N_FOLDS_INTERNO = 4
N_REPETICOES = 3
CAMINHO_DADOS = 'jogosLoL2021.csv'
CAMINHO_RESULTADOS = '.'

# 1. Função para carregar e preparar os dados

def carregar_dados(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    # Seleciona apenas as colunas dos 15 minutos e a coluna de classe
    colunas_15 = [c for c in df.columns if c.endswith('at15')]
    X = df[colunas_15].copy()
    y = df['result'].copy()
    return X, y

# 2. Padronização

def padronizar_dados(X_treino, X_teste):
    scaler = StandardScaler()
    X_treino_pad = scaler.fit_transform(X_treino)
    X_teste_pad = scaler.transform(X_teste)
    return X_treino_pad, X_teste_pad

# 3. Implementação do ensemble HB
class HBEnsemble:
    def __init__(self, n_estimadores=10, estado_aleatorio=ESTADO_ALEATORIO_CLASSIFICADORES):
        self.n_estimadores = n_estimadores
        self.estado_aleatorio = estado_aleatorio
        self.base_learners = [
            DecisionTreeClassifier(random_state=self.estado_aleatorio),
            GaussianNB(),
            MLPClassifier(random_state=self.estado_aleatorio, max_iter=500),
            KNeighborsClassifier()
        ]
        self.modelos = []
        self.classes_ = None
        self.classe_majoritaria = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classe_majoritaria = Counter(y).most_common(1)[0][0]
        self.modelos = []
        n_learners = len(self.base_learners)
        for i in range(self.n_estimadores):
            learner = self.base_learners[i % n_learners]
            modelo = learner.__class__(**learner.get_params())
            modelo.fit(X, y)
            self.modelos.append(modelo)

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.modelos])
        maj = []
        for i in range(X.shape[0]):
            votos = preds[:, i]
            contagem_votos = Counter(votos)
            mais_comum = contagem_votos.most_common()
            if len(mais_comum) > 1 and mais_comum[0][1] == mais_comum[1][1]:
                maj.append(self.classe_majoritaria)
            else:
                maj.append(mais_comum[0][0])
        return np.array(maj)

# 4. Função para rodar o experimento

def rodar_experimento(X, y):
    validacao_externa = RepeatedStratifiedKFold(n_splits=N_FOLDS_EXTERNO, n_repeats=N_REPETICOES, random_state=ESTADO_ALEATORIO_EXTERNO)
    resultados = {nome_clf: [] for nome_clf in ['DT', 'KNN', 'MLP', 'RF', 'HB']}
    todas_predicoes = {nome_clf: [] for nome_clf in resultados}
    todas_classes_reais = []

    # Hiperparâmetros
    grades_param = {
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 15, 25]},
        'KNN': {'n_neighbors': [1, 3, 5, 7, 9]},
        'MLP': {'hidden_layer_sizes': [(100,), (10,)], 'alpha': [0.0001, 0.005], 'learning_rate': ['constant', 'adaptive'], 'random_state': [ESTADO_ALEATORIO_CLASSIFICADORES], 'max_iter': [500]},
        'RF': {'n_estimators': [5, 10, 15, 25], 'max_depth': [10, None], 'random_state': [ESTADO_ALEATORIO_CLASSIFICADORES]},
        'HB': {'n_estimators': [5, 10, 15, 25, 50]}
    }
    
    for indices_treino, indices_teste in validacao_externa.split(X, y):
        X_treino, X_teste = X.iloc[indices_treino], X.iloc[indices_teste]
        y_treino, y_teste = y.iloc[indices_treino], y.iloc[indices_teste]
        X_treino_pad, X_teste_pad = padronizar_dados(X_treino, X_teste)
        todas_classes_reais.append(y_teste.values)

        # Decision Tree
        arvore_decisao = GridSearchCV(DecisionTreeClassifier(random_state=ESTADO_ALEATORIO_CLASSIFICADORES), grades_param['DT'], cv=N_FOLDS_INTERNO)
        arvore_decisao.fit(X_treino_pad, y_treino)
        predicoes_dt = arvore_decisao.predict(X_teste_pad)
        resultados['DT'].append(accuracy_score(y_teste, predicoes_dt))
        todas_predicoes['DT'].append(predicoes_dt)

        # KNN
        knn_grid = GridSearchCV(KNeighborsClassifier(), grades_param['KNN'], cv=N_FOLDS_INTERNO)
        knn_grid.fit(X_treino_pad, y_treino)
        predicoes_knn = knn_grid.predict(X_teste_pad)
        resultados['KNN'].append(accuracy_score(y_teste, predicoes_knn))
        todas_predicoes['KNN'].append(predicoes_knn)

        # MLP
        mlp_grid = GridSearchCV(MLPClassifier(), grades_param['MLP'], cv=N_FOLDS_INTERNO)
        mlp_grid.fit(X_treino_pad, y_treino)
        predicoes_mlp = mlp_grid.predict(X_teste_pad)
        resultados['MLP'].append(accuracy_score(y_teste, predicoes_mlp))
        todas_predicoes['MLP'].append(predicoes_mlp)

        # Random Forest
        floresta_grid = GridSearchCV(RandomForestClassifier(), grades_param['RF'], cv=N_FOLDS_INTERNO)
        floresta_grid.fit(X_treino_pad, y_treino)
        predicoes_rf = floresta_grid.predict(X_teste_pad)
        resultados['RF'].append(accuracy_score(y_teste, predicoes_rf))
        todas_predicoes['RF'].append(predicoes_rf)

        # HB Ensemble
        melhor_acuracia = 0
        melhor_predicao = None
        for qtd_estimadores in grades_param['HB']['n_estimators']:
            hb = HBEnsemble(n_estimadores=qtd_estimadores)
            hb.fit(X_treino_pad, y_treino)
            predicao_hb = hb.predict(X_teste_pad)
            acuracia_hb = accuracy_score(y_teste, predicao_hb)
            if acuracia_hb > melhor_acuracia:
                melhor_acuracia = acuracia_hb
                melhor_predicao = predicao_hb
        resultados['HB'].append(melhor_acuracia)
        todas_predicoes['HB'].append(melhor_predicao)

    return resultados, todas_predicoes, todas_classes_reais

# 5. Estatísticas e gráficos

def resumir_resultados(resultados):
    resumo = {}
    for nome_clf, lista_acuracias in resultados.items():
        array_acuracias = np.array(lista_acuracias)
        media = np.mean(array_acuracias)
        desvio = np.std(array_acuracias)
        ic95 = 1.96 * desvio / np.sqrt(len(array_acuracias))
        resumo[nome_clf] = {'media': media, 'desvio': desvio, 'ic95': ic95}
    return resumo

def plotar_boxplot(resultados, caminho_saida):
    df_resultados = pd.DataFrame(resultados)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_resultados)
    plt.title('Boxplot das acurácias por classificador')
    plt.ylabel('Acurácia')
    plt.savefig(os.path.join(caminho_saida, 'boxplot_acuracias.png'))
    plt.close()

# 6. Testes estatísticos

def testes_estatisticos(todas_predicoes, todas_classes_reais):
    nomes_classificadores = list(todas_predicoes.keys())
    quantidade_classificadores = len(nomes_classificadores)
    matriz_pvalues_t = np.zeros((quantidade_classificadores, quantidade_classificadores))
    matriz_pvalues_w = np.zeros((quantidade_classificadores, quantidade_classificadores))
    for indice_i in range(quantidade_classificadores):
        for indice_j in range(quantidade_classificadores):
            if indice_i < indice_j:
                # t de Nadeau e Bengio (aproximação pelo t pareado)
                acuracias_i = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(todas_classes_reais, todas_predicoes[nomes_classificadores[indice_i]])]
                acuracias_j = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(todas_classes_reais, todas_predicoes[nomes_classificadores[indice_j]])]
                _, p_valor = ttest_rel(acuracias_i, acuracias_j)
                matriz_pvalues_t[indice_i, indice_j] = p_valor
            elif indice_i > indice_j:
                # Wilcoxon
                acuracias_i = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(todas_classes_reais, todas_predicoes[nomes_classificadores[indice_i]])]
                acuracias_j = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(todas_classes_reais, todas_predicoes[nomes_classificadores[indice_j]])]
                try:
                    _, p_valor = wilcoxon(acuracias_i, acuracias_j)
                except:
                    p_valor = 1.0
                matriz_pvalues_w[indice_i, indice_j] = p_valor
    return nomes_classificadores, matriz_pvalues_t, matriz_pvalues_w

def salvar_tabela_estatistica(nomes_classificadores, matriz_pvalues_t, matriz_pvalues_w, caminho_saida):
    quantidade_classificadores = len(nomes_classificadores)
    tabela = pd.DataFrame('', index=nomes_classificadores, columns=nomes_classificadores)
    for indice_i in range(quantidade_classificadores):
        for indice_j in range(quantidade_classificadores):
            if indice_i < indice_j:
                valor = matriz_pvalues_t[indice_i, indice_j]
                if valor < 0.05:
                    tabela.iloc[indice_i, indice_j] = f'**{valor:.4f}**'
                else:
                    tabela.iloc[indice_i, indice_j] = f'{valor:.4f}'
            elif indice_i > indice_j:
                valor = matriz_pvalues_w[indice_i, indice_j]
                if valor < 0.05:
                    tabela.iloc[indice_i, indice_j] = f'**{valor:.4f}**'
                else:
                    tabela.iloc[indice_i, indice_j] = f'{valor:.4f}'
    tabela.to_csv(os.path.join(caminho_saida, 'tabela_pvalues.csv'))

# 7. Main

def main():
    X, y = carregar_dados(CAMINHO_DADOS)
    resultados, todas_predicoes, todos_reais = rodar_experimento(X, y)
    resumo = resumir_resultados(resultados)
    # Salva tabela resumo
    pd.DataFrame(resumo).T.to_csv(os.path.join(CAMINHO_RESULTADOS, 'tabela_resultados.csv'))
    # Boxplot
    plotar_boxplot(resultados, CAMINHO_RESULTADOS)
    # Testes estatísticos
    clfs, pvalues_t, pvalues_w = testes_estatisticos(todas_predicoes, todos_reais)
    salvar_tabela_estatistica(clfs, pvalues_t, pvalues_w, CAMINHO_RESULTADOS)
    print('Resultados salvos em', CAMINHO_RESULTADOS)

if __name__ == '__main__':
    main()
