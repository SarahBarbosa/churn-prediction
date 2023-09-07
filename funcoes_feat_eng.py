import pandas as pd
import numpy as np

def criar_segmentos(dados):
    """
    Cria segmentos com base na duração do cliente e nas mensalidades.

    Parâmetros:
        - dados (DataFrame): DataFrame contendo os dados dos clientes.
    """
    # Segmentos anuais para Tenure
    intervalos = [0, 12, 24, 36, 48, 60, 72]
    rotulos = ['0-1 Ano', '1-2 Anos', '2-3 Anos', '3-4 Anos', '4-5 Anos', '5-6 Anos']
    
    dados['Tenure_anual'] = pd.cut(dados['Tenure'], bins = intervalos, labels = rotulos)

    # Segmentos para clientes de acordo com os gastos mensais e total
    rotulos = ['Baixo', 'Moderado', 'Alto', 'Muito Alto']
    dados['Charges_monthly_categories'] = pd.cut(dados['Charges_monthly'], bins = 4, labels = rotulos)
    dados['Charges_total_categories'] = pd.cut(dados['Charges_total'], bins = 4, labels = rotulos)

def criar_indicador_internet_suporte(dados):
    """
    Cria um indicador de suporte com base em várias.

    Parâmetros:
        - dados (DataFrame): DataFrame contendo os dados dos clientes.
    """
    dados['No_internet_support_service'] = np.where((dados['Onlinesecurity'] != 1) | 
    (dados['Onlinebackup'] != 1) | (dados['Deviceprotection'] != 1), 1, 0)

def criar_indicador_todos_servicos_internet(dados):
    """
    Cria um indicador para o número de serviços de internet que atendem a 
    determinadas condições.

    Parâmetros:
        - dados (DataFrame): DataFrame contendo os dados dos clientes.
    """
    servicos = ['Onlinesecurity', 'Onlinebackup', 'Deviceprotection',
                'Techsupport', 'Streamingtv', 'Streamingmovies']

    for servico in servicos:
        dados[f'{servico}_temp'] = (dados[servico] == 1) & (dados['Nophoneservice'] == 0) & (dados['Nointernetservice'] == 0)

    dados['Num_internet_services'] = dados[[f'{servico}_temp' for servico in servicos]].sum(axis=1)
    dados.drop(columns=[f'{servico}_temp' for servico in servicos], inplace=True)

def calcular_estatisticas_gastos(dados):
    """
    Calcula estatísticas adicionais dos clientes.

    Parâmetros:
        - dados (DataFrame): DataFrame contendo os dados dos clientes.
    """
    dados['Avg_charges_monthly'] = dados['Charges_total'] / (dados['Tenure'])
    dados['Avg_charges_services'] = dados['Charges_total'] / (dados['Num_internet_services'] + 1)