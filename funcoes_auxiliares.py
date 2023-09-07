import pandas as pd

def descrever_colunas(dados):
    """
    Esta função recebe um DataFrame e exibe informações resumidas sobre suas 
    colunas, incluindo o tipo de dados e, se houver mais de 4 valores únicos, 
    a contagem de valores únicos.

    Parâmetros:
    - dados (pd.DataFrame): O DataFrame contendo os dados a serem inspecionados.
    """
    for col in dados.columns:
        tipo_de_dados = dados[col].dtype
        num_valores_unicos = dados[col].nunique()

        if num_valores_unicos > 4:
            print(f'Coluna {col}: {tipo_de_dados}, {num_valores_unicos}/{len(dados)} valores únicos')
        else:
            valores_unicos = ', '.join(map(str, dados[col].unique()))
            print(f'Coluna {col}: {tipo_de_dados}, Valores Únicos: {valores_unicos}')


def descrever_dados_faltantes(dados, coluna):
    """
    Esta função recebe um DataFrame e o nome de uma coluna e exibe informações 
    sobre os dados faltantes nessa coluna.

    Parâmetros:
    - dados (pd.DataFrame): O DataFrame contendo os dados a serem inspecionados.
    - coluna (str): O nome da coluna a ser analisada.
    """
    dados_faltantes = dados[coluna].isnull().sum()
    porcentagem_faltantes = round((dados_faltantes / len(dados)) * 100, 2)

    print(f'Dados faltantes na coluna {coluna}: {dados_faltantes}. \
     Isso representa {porcentagem_faltantes}% do total.')

def preprocessamento_binarizacao(df, colunas_sim_nao, colunas_multiclasses, colunas_mistas):
    """
    Realiza o pré-processamento de um DataFrame, convertendo colunas categóricas 
    em formatos binários ou one-hot encoding.

    Parâmetros:
    - df (pd.DataFrame): O DataFrame que contém os dados a serem pré-processados.
    - colunas_sim_nao (list): Uma lista de nomes de colunas que contêm valores 
    'Yes' e 'No' a serem binarizados (0 ou 1).
    - colunas_multiclasses (list): Uma lista de nomes de colunas com categorias 
    múltiplas a serem transformadas em one-hot encoding.
    - colunas_mistas (list): Uma lista de nomes de colunas que podem conter 
    valores 'Yes', 'No', 'No internet service' e 'No phone service'.

    Retorna:
    - pd.DataFrame: O DataFrame pré-processado com as colunas convertidas.
    """
    # Binariza colunas com valores 'Yes' e 'No'
    df[colunas_sim_nao] = df[colunas_sim_nao].replace({'Yes': 1, 'No': 0})
    
    # Converte a coluna 'customer.gender' em binário (Female: 1, Male: 0)
    df['customer.gender'] = df['customer.gender'].replace({'Female': 1, 'Male': 0})

    # Realiza one-hot encoding nas colunas de multiclasses
    df = pd.get_dummies(df, columns=colunas_multiclasses, dtype=int)

    # Função para criar colunas de serviço
    def criando_coluna_sem_servico(row):
        no_phone_service = 1 if 'No phone service' in row.values else 0
        no_internet_service = 1 if 'No internet service' in row.values else 0
        return pd.Series({'NoPhoneService': no_phone_service, 
        'NoInternetService': no_internet_service})

    # Aplica a função de criação de colunas de serviço
    colunas_sem_servico = df[colunas_mistas].apply(criando_coluna_sem_servico, axis=1)

    # Binariza colunas mistas
    df[colunas_mistas] = df[colunas_mistas].replace({'Yes': 1, 'No': 0, 
                                                     'No internet service': 0, 
                                                     'No phone service': 0})

    # Concatena as colunas de serviço ao DataFrame
    return pd.concat([df, colunas_sem_servico], axis=1)

def renomeia_coluna(nome_coluna):
    """
    Esta função recebe o nome de uma coluna em formato de string e realiza as 
    seguintes operações:
    
    1. Divide o nome da coluna usando o caractere '.' como separador.
    2. Pega a última parte após o último ponto (se houver pontos no nome).
    3. Converte a primeira letra da parte final para maiúscula, substitui 
    espaços por '_' e tira o parêntese.

    Parâmetros:
    - nome_coluna (str): O nome da coluna que deseja renomear.
    """
    primeiro_ponto = nome_coluna.find('.')
    if primeiro_ponto != -1:
        novo_nome = nome_coluna[primeiro_ponto+1:]
        novo_nome = novo_nome.replace('.', '_')
    else:
        novo_nome = nome_coluna
    novo_nome = novo_nome.capitalize().replace(' ', '_').replace(')', '').replace('(', '')
    return novo_nome