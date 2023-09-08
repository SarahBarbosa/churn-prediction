import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class ProcessadorDadosCustomizado(BaseEstimator, TransformerMixin):
    """
    Classe para pré-processamento de dados personalizado.
    
    Parâmetros:
    adicionar_todas_modificacoes (bool): Indica se todas as modificações 
    devem ser aplicadas ao conjunto de dados.
    """

    def __init__(self, adicionar_todas_modificacoes=True):
        self.adicionar_todas_modificacoes = adicionar_todas_modificacoes

    def fit(self, X):
        """
        Método de treinamento do estimador.
        
        Parâmetros:
        X (array-like): Os dados de entrada.
        """
        return self

    def transform(self, X):
        """
        Método de transformação dos dados.

        Parâmetros:
        X (array-like): Os dados de entrada.
        """
        data = pd.DataFrame(X)

        if self.adicionar_todas_modificacoes:
            self.criar_segmentos(data)
            self.criar_indicador_todos_servicos_internet(data)
            self.calcular_estatisticas_despesas(data)

        return  data

    def criar_segmentos(self, data):
        """
        Cria colunas de segmentos com base na coluna 'Tenure'.

        Parâmetros:
        data (DataFrame): O DataFrame de dados.
        """
        intervalos = [0, 12, 24, 36, 48, 60, 72]
        rotulos = ['0-1 Year', '1-2 Year', '2-3 Year', '3-4 Year', '4-5 Year', '5-6 Year']
        data['Tenure_category'] = pd.cut(data['Tenure'], bins=intervalos, 
                                         labels=rotulos)

        rotulos = ['Low', 'Moderate', 'High', 'Very High']
        data['Charges_monthly_cat'] = pd.cut(data['Charges_monthly'], bins = 4, labels=rotulos)
        data['Charges_total_cat'] = pd.cut(data['Charges_total'], bins = 4, labels=rotulos)

    def criar_indicador_todos_servicos_internet(self, data):
        """
        Cria uma coluna indicadora para todos os serviços de internet.

        Parâmetros:
        data (DataFrame): O DataFrame de dados.

        Retorna:
        None
        """
        servicos = ['Onlinesecurity', 'Onlinebackup', 'Deviceprotection',
                    'Techsupport', 'Streamingtv', 'Streamingmovies']

        for servico in servicos:
            data[f'{servico}_temp'] = (data[servico] == 1) & (data['Nointernetservice'] == 0)

        data['Num_internet_service'] = data[[f'{servico}_temp' for servico in servicos]].sum(axis=1)
        data.drop(columns=[f'{servico}_temp' for servico in servicos], inplace=True)

    def calcular_estatisticas_despesas(self, data):
        """
        Calcula estatísticas de despesas e cria colunas correspondentes.

        Parâmetros:
        data (DataFrame): O DataFrame de dados.
        """
        data['charge_per_internet_service'] = data['Charges_monthly'] / (data['Num_internet_service'] + 1)
