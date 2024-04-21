import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as PipelineIMB

from typing import Union, List, Tuple, Dict, Callable, Any


def info_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera um DataFrame com informações sobre as colunas do DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        O DataFrame a ser analisado.

    Returns:
    --------
    pd.DataFrame
        Um DataFrame contendo informações sobre as colunas do DataFrame fornecido.
    """
    col_types: pd.Series = df.dtypes
    unique_values: pd.DataFrame = df.apply(lambda col: col.unique())
    info_df: pd.DataFrame = pd.DataFrame({'Tipo': col_types, 'Valores Únicos': unique_values}, index=df.columns)

    return info_df

class Tratamentos:
    """
    Classe para realizar tratamentos iniciais.
    """
    def __init__(self, dados: pd.DataFrame) -> None:
        """
        Inicializa a classe Tratamentos.

        Parameters:
        -----------
        dados : pd.DataFrame
            O DataFrame a ser tratado.
        """
        self.dados: pd.DataFrame = dados

    def string_to_float(self, colunas: Union[str, List[str]]) -> None:
        """
        Converte colunas de strings para float.

        Parameters:
        -----------
        colunas : str or List[str]
            Nome da coluna ou lista de nomes das colunas a serem convertidas.
        """
        self.dados[colunas] = pd.to_numeric(self.dados[colunas], errors='coerce')

    def remove_ausentes(self, colunas: str, string: bool = True) -> pd.DataFrame:
        """
        Remove linhas com valores ausentes em uma coluna específica.

        Parameters:
        -----------
        colunas : str
            Nome da coluna a ser considerada.
        string : bool, optional
            Se True, remove linhas com strings vazias, caso contrário, remove linhas com NaN.

        Returns:
        --------
        pd.DataFrame
            O DataFrame após a remoção das linhas.
        """
        if string:
            self.dados = self.dados.loc[~(self.dados[colunas] == '')].reset_index(drop=True)
        else:
            self.dados.dropna(subset=[colunas], inplace=True)
        return self.dados

    def remove_colunas(self, colunas: Union[str, List[str]]) -> pd.DataFrame:
        """
        Remove colunas do DataFrame.

        Parameters:
        -----------
        colunas : str or List[str]
            Nome da coluna ou lista de nomes das colunas a serem removidas.

        Returns:
        --------
        pd.DataFrame
            O DataFrame após a remoção das colunas.
        """
        self.dados = self.dados.drop(colunas, axis=1)
        return self.dados

def adicionar_estatisticas(ax: plt.Axes, dados: pd.DataFrame, metrica: str, texto_y: float, cor: str) -> None:
    """
    Adiciona estatísticas resumidas a um subplot.

    Parameters:
    -----------
    ax : plt.Axes
        O subplot onde as estatísticas serão adicionadas.
    dados : pd.DataFrame
        O DataFrame contendo os dados.
    metrica : str
        A métrica para a qual as estatísticas serão calculadas.
    texto_y : float
        A posição y do texto.
    cor : str
        A cor do texto.
    """
    val: pd.DataFrame = dados.groupby('Churn')[metrica].agg(['mean', 'median', 'min', 'max'])

    texto: str = f'Média: {val["mean"].iloc[0]:.2f} | {val["mean"].iloc[1]:.2f}\n'
    texto += f'Mediana: {val["median"].iloc[0]:.2f} | {val["median"].iloc[1]:.2f}\n'
    texto += f'Mínimo: {val["min"].iloc[0]:.2f} | {val["min"].iloc[1]:.2f}\n'
    texto += f'Máximo: {val["max"].iloc[0]:.2f} | {val["max"].iloc[1]:.2f}'

    ax.text(0.5, texto_y, texto, transform=ax.transAxes, fontsize=12, color=cor, ha='center')

def renomear_coluna(nome_coluna: str) -> str:
    """
    Esta função recebe o nome de uma coluna em formato de string e realiza as 
    seguintes operações:
    
    1. Divide o nome da coluna usando o caractere '.' como separador.
    2. Pega a última parte após o último ponto (se houver pontos no nome).
    3. Converte a primeira letra da parte final para maiúscula, substitui 
    espaços por '_' e tira o parêntese.

    Parameters:
    -----------
    nome_coluna : str
        O nome da coluna.

    Returns:
    --------
    str
        O nome da coluna após as operações de formatação.
    """
    primeiro_ponto: int = nome_coluna.find('.')
    if primeiro_ponto != -1:
        novo_nome: str = nome_coluna[primeiro_ponto + 1:]
        novo_nome = novo_nome.replace('.', '_')
    else:
        novo_nome = nome_coluna
    
    novo_nome = novo_nome.capitalize().replace(' ', '_').replace(')', '').replace('(', '')
    return novo_nome

def TotalServiceTransformer(X: pd.DataFrame) -> None:
    """
    A função calcula o número total de serviços contratados para cada cliente com base nas 
    colunas especificadas no lista 'servicos'. A função considera um serviço contratado se 
    o valor na coluna correspondente for igual a 'Yes' e conta quantos serviços estão ativos 
    para cada cliente.

    Parameters:
    -----------
    X : pd.DataFrame
        O DataFrame contendo os dados dos clientes.
    """
    servicos: List[str] = ['Multiplelines', 'Onlinesecurity', 'Onlinebackup',
                'Deviceprotection', 'Techsupport', 'Streamingtv',
                'Streamingmovies']
    contador_sim = lambda col: col.apply(lambda x: 1 if x == 'Yes' else 0)
    X['TotalServices'] = X[servicos].apply(contador_sim, axis=1).sum(axis=1)

def cross_validation_models_set(abordagem: str, preprocessor: Union[Callable, List[Tuple[str, Callable]]], 
                                X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> pd.DataFrame:
    """
    Realiza uma validação cruzada para avaliar o desempenho de diferentes modelos de classificação 
    usando diferentes abordagens de tratamento de desequilíbrio de classe.

    Parameters:
    -----------
    abordagem : str
        A abordagem usada para tratar o desequilíbrio de classe, pode ser 'Oversampling', 
        'Undersampling' ou 'Default'.
    preprocessor : Callable or List[Tuple[str, Callable]]
        O pré-processador a ser aplicado aos dados ou lista de tuplas com nome do pré-processador e função.
    X_train : pd.DataFrame
        Os dados de treinamento.
    y_train : pd.Series
        Os rótulos de treinamento.
    random_state : int, optional
        O estado aleatório para reprodutibilidade.

    Returns:
    --------
    pd.DataFrame
        Um DataFrame contendo os resultados da validação cruzada para diferentes modelos de classificação.
    """

    classifiers: Dict[str, Any] = {
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

def hyperparameter_optimization(resampler: Union[SMOTE, TomekLinks, None], 
                                preprocessor: Union[Callable, List[Tuple[str, Callable]]], 
                                X_train: pd.DataFrame, y_train: pd.Series, 
                                modelos: Dict[str, Tuple[Any, Dict[str, Union[List[int], List[float]]]]]) -> pd.DataFrame:
    """
    Otimiza hiperparâmetros para modelos de classificação usando GridSearchCV.
    
    Parameters:
    -----------
    resampler : SMOTE, TomekLinks, or None
        O método de reamostragem a ser aplicado.
    preprocessor : Callable or List[Tuple[str, Callable]]
        O pré-processador a ser aplicado aos dados ou lista de tuplas com nome do pré-processador e função.
    X_train : pd.DataFrame
        Os dados de treinamento.
    y_train : pd.Series
        Os rótulos de treinamento.
    modelos : Dict[str, Tuple[Any, Dict[str, Union[List[int], List[float]]]]]
        Um dicionário contendo os modelos de classificação e os hiperparâmetros a serem otimizados.

    Returns:
    --------
    pd.DataFrame
        Um DataFrame contendo os melhores scores de F1 e Recall para cada modelo.
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
