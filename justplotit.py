import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

def frequencia_churn(dados, palette = 'novexus'):
  """
  Plota a frequência de Churn.

  Parâmetros:
  - data (DataFrame): O conjunto de dados que contém a coluna 'Churn'.
  - palette (str ou lista de cores, opcional): Paleta de cores a ser usada no 
  gráfico. Default = 'novexus'
  """
  if palette == 'novexus':
    paleta = ['#171821', '#872b95', '#ff7131', '#fe3d67']
  
  plt.figure(figsize=(10, 5))
  ax = sns.countplot(x='Churn', data=dados, palette=paleta, alpha=.8)
  sns.despine(right=True, top=True, left=True)
  ax.set(ylabel=None)
  ax.tick_params(left=False)
  ax.set(yticklabels=[])

  for p in ax.patches:
      x = p.get_x() + p.get_width() / 2
      y = p.get_height()
      ax.annotate(f'{y}\n({y/len(dados)*100:.2f}%)\n', (x, y),
                  ha='center', va='bottom', color='gray')

  plt.title('Frequência Churn', y=1.2)
  plt.show()

def countplots(data, coluna_hue, grupos, paleta, figsize=(12, 8)):
  """
  Gera subplots de gráficos de barras com contagem para várias colunas 
  categóricas.

  Parâmetros:
  - data (pd.DataFrame): O DataFrame contendo os dados a serem visualizados.
  - coluna_hue (str): O nome da coluna que será usada para a diferenciação das 
  barras por cores (hue).
  - grupos (list): Uma lista de nomes de colunas categóricas a serem 
  visualizadas.
  - paleta (str ou list): Uma paleta de cores para a diferenciação das barras.
  Pode ser uma string de paleta ou uma lista de cores.
  - figsize (tuple): Tamanho da figura (largura, altura).
  """
  num_grupos = len(grupos)
  cols = 2
  linhas = int(np.ceil(num_grupos / cols))

  fig, axes = plt.subplots(nrows=linhas, ncols=cols, figsize=figsize)
  axes = axes.flatten()  # Transforma a matriz de eixos em uma matriz 1D

  for i, grupo in enumerate(grupos):
      if i < num_grupos:
          eixo = axes[i]  # Acessa o eixo diretamente da matriz 1D
          sns.countplot(x=grupo,
                        data=data,
                        hue=coluna_hue,
                        palette=paleta,
                        alpha=0.8,
                        ax=eixo)
          sns.despine(right=True, top=True, left = True)
          eixo.set(ylabel=None)
          eixo.tick_params(left=False)
          eixo.set(yticklabels=[])
          eixo.legend([], [], frameon=False)
          eixo.set_xlabel(grupo.split('.')[-1])
          eixo.set_ylabel('')

          # Adiciona o valor da frequência em cima de cada barra
          for p in eixo.patches:
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            eixo.annotate(f'{y}\n({y/len(data)*100:.2f}%)\n', (x, y),
            ha='center', va='bottom', color='gray')

  # Remove quaisquer subplots vazios adicionais
  for i in range(num_grupos, linhas * cols):
      fig.delaxes(axes[i])

  # Cria uma legenda fora dos subplots
  handles, labels = eixo.get_legend_handles_labels()
  legenda = fig.legend(handles,
                        labels,
                        loc='upper center',
                        bbox_to_anchor=(0.5, 1.13),
                        ncol=2)
  legenda.set_title('Churn')

  plt.tight_layout()

def histogramas(dados, features_num, paleta,):
  """
  Plota histogramas para colunas numéricas com base na variável 'Churn'.

  Args:
      dados (DataFrame): O DataFrame contendo os dados.
      features_num (DataFrame): O DataFrame contendo as colunas numéricas.
      paleta (list): Uma paleta de cores para os gráficos.
      legend_labels (list): Uma lista de rótulos para a legenda.
      legend_title (str): O título da legenda.
  """
  fig, axes = plt.subplots(1, len(features_num.columns), figsize=(13, 5), sharey=True)

  for i, col in enumerate(features_num.columns):
      sns.histplot(data=dados, x=dados[col], hue='Churn', ax=axes[i], palette=paleta[:2], kde=True)
      sns.despine(right=True, top=True)
      col_name = ' '.join(col.split('.')[1:])
      axes[i].set_xlabel(col_name)
      axes[i].legend([], [], frameon=False)
      axes[i].set_ylabel('Frequência')

  legend_labels = ['No', 'Yes']
  legend_colors = [paleta[0], paleta[1]]
  custom_legend = dict(zip(legend_labels, legend_colors))

  handles = [plt.Rectangle((0, 0), 1, 1, color=custom_legend[label], label=label) for label in legend_labels]
  legend = fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1), title='Churn')

  plt.tight_layout()
  plt.show()

def adicionar_estatisticas(ax, dados, metrica, texto_y, cor):
  """
  Adiciona estatísticas resumidas a um subplot.

  Parâmetros:
  - ax (matplotlib.axes._subplots.AxesSubplot): O subplot ao qual as 
  estatísticas serão adicionadas.
  - dados (pd.DataFrame): O DataFrame contendo os dados a serem analisados.
  - metrica (str): A métrica para a qual as estatísticas serão calculadas 
  (por exemplo, 'total_charges').
  - texto_y (float): A posição vertical do texto no subplot.
  - cor (str): A cor do texto.
  """
  valores_estatisticos = dados.groupby('Churn')[metrica].agg(['mean', 'median', 'min', 'max'])

  texto = f'Média: {valores_estatisticos["mean"][0]:.2f} | {valores_estatisticos["mean"][1]:.2f}\n'
  texto += f'Mediana: {valores_estatisticos["median"][0]:.2f} | {valores_estatisticos["median"][1]:.2f}\n'
  texto += f'Mínimo: {valores_estatisticos["min"][0]:.2f} | {valores_estatisticos["min"][1]:.2f}\n'
  texto += f'Máximo: {valores_estatisticos["max"][0]:.2f} | {valores_estatisticos["max"][1]:.2f}'

  ax.text(0.5, texto_y, texto, transform=ax.transAxes, fontsize=12, color=cor, ha='center')

def boxplots(dados, features_num, paleta):
    """
    Plota boxplots para colunas numéricas com base na variável 'Churn'.

    Args:
        dados (DataFrame): O DataFrame contendo os dados.
        features_num (DataFrame): O DataFrame contendo as colunas numéricas.
        paleta (list): Uma paleta de cores para os gráficos.
    """
    fig, axes = plt.subplots(1, len(features_num.columns), figsize=(13, 7))

    for i, col in enumerate(features_num.columns):
        boxplot = sns.boxplot(x='Churn', y=col, data=dados, ax=axes[i], palette=paleta[1:])
        for patch in boxplot.artists:
            patch.set_alpha(0.5)
        sns.despine(right=True, top=True)
        col_name = ' '.join(col.split('.')[1:])
        adicionar_estatisticas(axes[i], dados, col, -0.3, paleta[-1])
        axes[i].set_title(col_name)
        axes[i].set_ylabel('')

    plt.tight_layout()
    plt.show()

def feature_importance_comparacao(importancia_model_default, 
                                       importancia_model_oversampling, 
                                       nome_model_default, 
                                       nome_model_oversampling, resize=True):
    """
    Plota um gráfico de barras comparando a importância das features entre dois 
    modelos.

    Parâmetros:
    importancia_model_default (DataFrame): Um DataFrame contendo a importância 
    das features do modelo padrão.
    importancia_model_oversampling (DataFrame): Um DataFrame contendo a 
    importância das features do modelo com oversampling.
    nome_model_default (str): O nome do modelo padrão (será exibido no gráfico).
    nome_model_oversampling (str): O nome do modelo com oversampling 
    (será exibido no gráfico).
    resize (bool, opcional): Se True, a importância das features do modelo com 
    oversampling será redimensionada para variar entre 0 e 1.  Se False, a 
    importância será usada sem alteração. O padrão é True.
    """

    # Redimensiona a importância das features do modelo com oversampling, se necessário
    if resize:
        importancia_model_oversampling['Importância'] = importancia_model_oversampling['Importância'] / 100
    else:
        importancia_model_oversampling['Importância'] = importancia_model_oversampling['Importância']

    plt.figure(figsize=(8, 9))

    ax = sns.barplot(x='Importância', y='Feature', data=importancia_model_default, 
                    label=f'{nome_model_default} (Default Sample)', alpha=0.5, color= '#872b95')
    ax = sns.barplot(x='Importância', y='Feature', data=importancia_model_oversampling, 
                    label=f'{nome_model_oversampling} (Oversampling)', alpha=0.5, color= '#fe3d67')
    sns.despine(right=True, top=True, bottom=True)
    plt.title(f'Feature Importance Comparison ({nome_model_default} vs. {nome_model_oversampling})')
    

    plt.legend()
    ax.set(ylabel=None)
    plt.show()

def confusion_matrix(conf_matrix, nome_modelo):
  labels = ['Não Churn', 'Churn']
  plt.figure(figsize=(8, 6))
  sns.set(font_scale=1.2)
  sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="flare", cbar=False,
              xticklabels=labels, yticklabels=labels)
  plt.xlabel('Valores previstos')
  plt.ylabel('Valores reais')
  plt.title(f'Matriz de confusão {nome_modelo}')
  plt.show()
