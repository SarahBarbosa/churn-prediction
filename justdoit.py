import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as PipelineIMB

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def info_dataframe(df):
    """
    Gera um DataFrame com informações sobre as colunas do DataFrame.
    """
    col_types = df.dtypes
    unique_values = df.apply(lambda col: col.unique())
    info_df = pd.DataFrame({'Tipo': col_types, 
                            'Valores Únicos': unique_values}, 
                            index=df.columns)

    return info_df

class Tratamentos:
    """
    Classe para realizar tratamentos iniciais.
    """
    def __init__(self, dados):
        self.dados = dados

    def string_to_float(self, colunas):
        self.dados[colunas] = pd.to_numeric(self.dados[colunas],
                                            errors='coerce')

    def remove_ausentes(self, colunas, string=True):
        if string == True:
            self.dados = self.dados.loc[~(
                self.dados[colunas] == '')].reset_index(drop=True)
        elif string == False:
            self.dados.dropna(subset=[colunas], inplace=True)
        return self.dados

    def remove_colunas(self, colunas):
        self.dados = self.dados.drop(colunas, axis=1)
        return self.dados

def adicionar_estatisticas(ax, dados, metrica, texto_y, cor):
  """
  Adiciona estatísticas resumidas a um subplot.
  """
  val = dados.groupby('Churn')[metrica].agg(['mean', 'median', 'min', 'max'])

  texto = f'Média: {val["mean"].iloc[0]:.2f} | {val["mean"].iloc[1]:.2f}\n'
  texto += f'Mediana: {val["median"].iloc[0]:.2f} | {val["median"].iloc[1]:.2f}\n'
  texto += f'Mínimo: {val["min"].iloc[0]:.2f} | {val["min"].iloc[1]:.2f}\n'
  texto += f'Máximo: {val["max"].iloc[0]:.2f} | {val["max"].iloc[1]:.2f}'

  ax.text(0.5, texto_y, texto, transform=ax.transAxes, fontsize=12, color=cor, ha='center')

def renomear_coluna(nome_coluna):
    """
  Esta função recebe o nome de uma coluna em formato de string e realiza as 
  seguintes operações:
  
  1. Divide o nome da coluna usando o caractere '.' como separador.
  2. Pega a última parte após o último ponto (se houver pontos no nome).
  3. Converte a primeira letra da parte final para maiúscula, substitui 
  espaços por '_' e tira o parêntese.
  """
    primeiro_ponto = nome_coluna.find('.')
    if primeiro_ponto != -1:
        novo_nome = nome_coluna[primeiro_ponto + 1:]
        novo_nome = novo_nome.replace('.', '_')
    else:
        novo_nome = nome_coluna
    
    novo_nome = novo_nome.capitalize().replace(' ', '_').replace(')', '').replace('(', '')
    return novo_nome


def TotalServiceTransformer(X):
    """
    A função calcula o número total de serviços contratados para cada cliente com base nas 
    colunas especificadas no lista 'servicos'. A função considera um serviço contratado se 
    o valor na coluna correspondente for igual a 'Yes' e conta quantos serviços estão ativos 
    para cada cliente.
    """
    servicos = ['Multiplelines', 'Onlinesecurity', 'Onlinebackup',
                'Deviceprotection', 'Techsupport', 'Streamingtv',
                'Streamingmovies']
    contador_sim = lambda col: col.apply(lambda x: 1 if x == 'Yes' else 0)
    X['TotalServices'] = X[servicos].apply(contador_sim, axis=1).sum(axis=1)


def cross_validation_models_set(abordagem, preprocessor, X_train, y_train, random_state=42):
    """
    Realiza uma validação cruzada para avaliar o desempenho de diferentes modelos de classificação 
    usando diferentes abordagens de tratamento de desequilíbrio de classe.

    abordagem (str): A abordagem usada para tratar o desequilíbrio de classe, pode ser 'Oversampling', 
    'Undersampling' ou 'Default'.
    """

    classifiers = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000),
        'K-Vizinhos Mais Próximos (KNN)': KNeighborsClassifier(),
        'Árvore de Decisão': DecisionTreeClassifier(random_state=random_state),
        'Floresta Randômica': RandomForestClassifier(random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Support Vector Machine': SVC(random_state=random_state)
    }

    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=random_state)
    
    resultados = []

    for nome, classificador in classifiers.items():
        if abordagem == 'Oversampling':
            resampler = SMOTE()
        elif abordagem == 'Undersampling':
            resampler = TomekLinks()
        else:
            resampler = None

        steps = [('Preprocessor', preprocessor)]
        if resampler:
            steps.append(('Resampler', resampler))
        steps.append(('Model', classificador))
        pipeline = PipelineIMB(steps)

        metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
        scores = [cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=metric, n_jobs = -1).mean() 
                  for metric in metrics]

        resultados.append([nome] + [round(score, 3) for score in scores])

    columns = ['Modelo', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    df_resultados = pd.DataFrame(resultados, columns=columns)

    return df_resultados


def hyperparameter_optimization(resampler, preprocessor, X_train, y_train, modelos):
    """
    Otimiza hiperparâmetros para modelos de classificação usando GridSearchCV.
    
    resampler: SMOTE(), TomekLinks() ou None.
    """
    best_score = []

    for model_name, (model, params) in modelos.items():
        pipe=PipelineIMB([('Preprocessor',preprocessor), ('Resampler', resampler), ('model',model)])
        grid_search = GridSearchCV(pipe, params, cv=2, scoring=['recall_macro', 'f1_macro'], refit='recall_macro' , n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        
        scores={'model':model_name,
                'F1_score':grid_search.cv_results_['mean_test_f1_macro'][grid_search.best_index_],
                'Recall':grid_search.cv_results_['mean_test_recall_macro'][grid_search.best_index_]}
        best_score.append(scores)

    return pd.DataFrame(best_score)