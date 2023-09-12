import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV


def descricao_colunas(dados):
    """
  Esta função recebe um DataFrame e exibe informações resumidas sobre suas 
  colunas, incluindo o tipo de dados e, se houver mais de 4 valores únicos, 
  a contagem de valores únicos.

  Parâmetros:
  - dados (pd.DataFrame): O DataFrame contendo os dados a serem inspecionados.
  """
    for col in dados.columns:
        tipo_dado = dados[col].dtype
        n_val_unico = dados[col].nunique()

        if n_val_unico > 4:
            print(
                f'# {col}: {tipo_dado}, {n_val_unico}/{len(dados)} valores únicos'
            )
        else:
            valores_unicos = ', '.join(map(str, dados[col].unique()))
            print(f'# {col}: {tipo_dado}, Valores Únicos: {valores_unicos}')


class Tratamentos:
    """
  Classe para realizar tratamentos iniciais.

  Args:
      dados (pandas.DataFrame): O DataFrame no qual os tratamentos serão 
      aplicados.

  Methods:
      string_to_float():
          Converte os valores das colunas especificadas para valores numéricos.

      remove_ausentes(string=True):
          Remove as linhas do DataFrame onde os valores da(s) coluna(s) 
          especificada(s) estão em branco (ou NaN).
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


def encoding_manual(dados, colunas_sim_nao, colunas_multiclasses,
                    colunas_mistas):
    """
  Realiza o pré-processamento de um DataFrame, convertendo colunas categóricas 
  em formatos one-hot encoding.

  Parâmetros:
  - dados (pd.DataFrame): O DataFrame que contém os dados a serem 
  pré-processados.
  - colunas_sim_nao (list): Uma lista de nomes de colunas que contêm valores 
  'Yes' e 'No' a serem binarizados (0 ou 1).
  - colunas_multiclasses (list): Uma lista de nomes de colunas com categorias 
  múltiplas a serem transformadas em one-hot encoding.
  - colunas_mistas (list): Uma lista de nomes de colunas que podem conter 
  valores 'Yes', 'No', 'No internet service' e 'No phone service'.
  """
    dados[colunas_sim_nao] = dados[colunas_sim_nao].replace({
        'Yes': 1,
        'No': 0
    })
    dados['customer.gender'] = dados['customer.gender'].replace({
        'Female': 1,
        'Male': 0
    })
    dados = pd.get_dummies(dados, columns=colunas_multiclasses, dtype=int)

    def criando_coluna_sem_servico(row):
        no_phone_service = 1 if 'No phone service' in row.values else 0
        no_internet_service = 1 if 'No internet service' in row.values else 0
        return pd.Series({
            'No_phone_service': no_phone_service,
            'No_internet_service': no_internet_service
        })

    colunas_sem_servico = dados[colunas_mistas].apply(
        criando_coluna_sem_servico, axis=1)

    dados[colunas_mistas] = dados[colunas_mistas].replace({
        'Yes':
        1,
        'No':
        0,
        'No internet service':
        0,
        'No phone service':
        0
    })

    return pd.concat([dados, colunas_sem_servico], axis=1)


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
    novo_nome = novo_nome.capitalize().replace(' ', '_').replace(')',
                                                                 '').replace(
                                                                     '(', '')
    return novo_nome


def pipeline_features(dados,
                      colunas_sim_nao,
                      colunas_multiclasses,
                      colunas_mistas,
                      binarizacao_manual=True,
                      add_novas_features=True,
                      drop_col_extra=True,
                      standarscaler=True,
                      renomeiar_col=True):
    """
    Um pipeline para pré-processamento de dados.

    Args:
        dados (DataFrame): O DataFrame contendo os dados.
        binarizacao_manual (bool): Se True, aplica binarização manual em 
        colunas específicas.
        add_novas_features (bool): Se True, cria novas features com base nos 
        dados existentes.
        drop_col_extra (bool): Se True, remove colunas adicionais especificadas.
        standarscaler (bool): Se True, aplica StandardScaler às colunas 
        numéricas.
        renomeiar_col (bool): Se True, renomeia as colunas do DataFrame.
    """

    if binarizacao_manual:
        dados = encoding_manual(dados, colunas_sim_nao, colunas_multiclasses,
                                colunas_mistas)
        dados = Tratamentos(dados).remove_colunas(
            'internet.InternetService_No')
        dados = Tratamentos(dados).remove_colunas('phone.PhoneService')

    if add_novas_features:
        dados = criar_novas_features(dados)

    if drop_col_extra:
        dados = Tratamentos(dados).remove_colunas('account.Charges.Total')

    if standarscaler:
        colunas_numericas = [
            col for col in dados.columns if dados[col].nunique() > 2
        ]
        scaler = StandardScaler()
        dados[colunas_numericas] = scaler.fit_transform(
            dados[colunas_numericas])

    if renomeiar_col:
        dados = dados.rename(columns=renomear_coluna)

    return dados.reset_index(drop=True)


def criar_novas_features(dados):
    """
    Cria novas features com base nos dados existentes.
    """
    servicos = [
        'internet.OnlineSecurity', 'internet.OnlineBackup',
        'internet.DeviceProtection', 'internet.TechSupport',
        'internet.StreamingTV', 'internet.StreamingMovies'
    ]

    for servico in servicos:
        dados[f'{servico}_temp'] = (dados[servico]
                                    == 1) & (dados['No_internet_service'] == 0)

    dados['Num_internet_service'] = dados[[
        f'{servico}_temp' for servico in servicos
    ]].sum(axis=1)
    dados.drop(columns=[f'{servico}_temp' for servico in servicos],
               inplace=True)

    dados['charge_per_internet_service'] = dados['account.Charges.Monthly'] / (
        dados['Num_internet_service'] + 1)

    return pd.get_dummies(dados)


def cross_validation_models_set(abordagem,
                                X_train,
                                y_train,
                                classificadores,
                                cv=3,
                                random_state=42):
    """
  Realiza validação cruzada para vários modelos de classificação.

  Parâmetros:
  abordagem (str): A abordagem utilizada para a validação cruzada ('oversampling' ou 'default').
  X_train (array-like): Conjunto de treinamento das features.
  y_train (array-like): Conjunto de treinamento das targets.
  classificadores (list): Uma lista de tuplas contendo o nome do modelo e o classificador.
  cv (int ou objeto cv): Número de dobras (folds) ou um objeto de validação cruzada StratifiedKFold.
  random_state (int): Seed para garantir a reprodutibilidade.

  Retorno:
  df_resultados (DataFrame): Um DataFrame contendo os resultados da validação cruzada para cada modelo.
  df_style (Styler): Um Styler pandas para destacar os resultados.
  """
    stratified_kfold = StratifiedKFold(n_splits=cv,
                                       shuffle=True,
                                       random_state=random_state)
    resultados = []

    for nome, classificador in classificadores:
        if abordagem == 'oversampling':
            roc_auc = cross_val_score(classificador,
                                      X_train,
                                      y_train,
                                      cv=cv,
                                      scoring='roc_auc').mean()
            accuracy = cross_val_score(classificador,
                                       X_train,
                                       y_train,
                                       cv=cv,
                                       scoring='accuracy').mean()
            precision = cross_val_score(classificador,
                                        X_train,
                                        y_train,
                                        cv=cv,
                                        scoring='precision').mean()
            recall = cross_val_score(classificador,
                                     X_train,
                                     y_train,
                                     cv=cv,
                                     scoring='recall').mean()
            f1 = cross_val_score(classificador,
                                 X_train,
                                 y_train,
                                 cv=cv,
                                 scoring='f1').mean()
        else:
            roc_auc = cross_val_score(classificador,
                                      X_train,
                                      y_train,
                                      cv=stratified_kfold,
                                      scoring='roc_auc').mean()
            accuracy = cross_val_score(classificador,
                                       X_train,
                                       y_train,
                                       cv=stratified_kfold,
                                       scoring='accuracy').mean()
            precision = cross_val_score(classificador,
                                        X_train,
                                        y_train,
                                        cv=stratified_kfold,
                                        scoring='precision').mean()
            recall = cross_val_score(classificador,
                                     X_train,
                                     y_train,
                                     cv=stratified_kfold,
                                     scoring='recall').mean()
            f1 = cross_val_score(classificador,
                                 X_train,
                                 y_train,
                                 cv=stratified_kfold,
                                 scoring='f1').mean()

        resultados.append([
            nome,
            round(roc_auc.mean(), 3),
            round(accuracy, 3),
            round(precision, 3),
            round(recall, 3),
            round(f1, 3)
        ])

    df_resultados = pd.DataFrame(resultados,
                                 columns=[
                                     'Modelo', 'ROC-AUC', 'Accuracy',
                                     'Precision', 'Recall', 'F1-Score'
                                 ])
    df_style = df_resultados.style.background_gradient(
        cmap='Spectral_r',
        subset=['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        axis=0)

    return df_resultados, df_style


def hyperparameter_optimization(X_train,
                                y_train,
                                modelos,
                                cv=3,
                                scoring=['accuracy', 'f1_macro']):
    """
  Otimiza hiperparâmetros para modelos de classificação usando GridSearchCV.

  Parâmetros:
  X_train (array-like): Conjunto de treinamento das features.
  y_train (array-like): Conjunto de treinamento das targets.
  modelos (list): Uma lista de tuplas contendo o nome do modelo, o modelo e os 
  parâmetros a serem otimizados.
  cv (int): Número de dobras (folds) para validação cruzada.
  scoring (list): Lista de métricas de avaliação a serem usadas para otimização.

  Retorno:
  resultados (DataFrame): Um DataFrame contendo os melhores resultados de 
  hiperparâmetros 
  para cada modelo.
  """
    melhor_score = []

    for nome_modelo, modelo, parametros in modelos:
        grid_search = GridSearchCV(modelo,
                                   parametros,
                                   cv=cv,
                                   scoring=scoring,
                                   refit='f1_macro',
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(
            f"# Melhores parâmetros para {nome_modelo}: {grid_search.best_params_}"
        )

        best_accuracy = grid_search.cv_results_['mean_test_accuracy'][
            grid_search.best_index_]
        best_f1_macro = grid_search.best_score_

        score = {
            'Modelo': nome_modelo,
            'F1 Macro Score': best_f1_macro,
            'Accuracy': best_accuracy
        }
        melhor_score.append(score)

    return pd.DataFrame(melhor_score)