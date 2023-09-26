import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import justdoit as jdi
from sklearn.metrics import confusion_matrix

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

def frequencia_churn(dados, palette='novexus'):
    """
    Plota a frequência de Churn.
    """
    if palette == 'novexus':
        paleta = ['#171821', '#872b95', '#ff7131', '#fe3d67']

    plt.figure(figsize=(10, 5))
    
    ax = sns.barplot(x=dados['Churn'].value_counts().index, 
                     y=dados['Churn'].value_counts(normalize=True), 
                     palette=paleta, 
                     alpha=0.8)
    
    sns.despine(right=True, top=True, left=True)
    ax.set(ylabel=None)
    ax.tick_params(left=False)
    ax.set(yticklabels=[])

    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(f'{y:.2%}\n', (x, y), ha='center', va='bottom', color='gray')

    plt.title('Frequência Churn', y=1.2);

def countplots(data, coluna_hue, grupos, paleta, figsize=(12, 8)):
  """
  Gera subplots de gráficos de barras com contagem para várias colunas 
  categóricas.
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

  plt.tight_layout();

def histogramas(dados, features_num, paleta,):
  """
  Plota histogramas para colunas numéricas com base na variável 'Churn'.
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

  plt.tight_layout();

def boxplots(dados, features_num, paleta):
    """
    Plota boxplots para colunas numéricas com base na variável 'Churn'.
    """
    _, axes = plt.subplots(1, len(features_num.columns), figsize=(13, 7))

    for i, col in enumerate(features_num.columns):
        sns.boxplot(x='Churn', y=col, data=dados, ax=axes[i], palette=paleta, boxprops=dict(alpha=.8))
        sns.despine(right=True, top=True)
        col_name = ' '.join(col.split('.')[1:])
        jdi.adicionar_estatisticas(axes[i], dados, col, -0.3, paleta[-1])
        axes[i].set_title(col_name)
        axes[i].set_ylabel('')

    plt.tight_layout();

def confusion_matrix(conf_matrix, nome_modelo):
  labels = ['Não Churn', 'Churn']
  plt.figure(figsize=(8, 6))
  sns.set(font_scale=1.2)
  sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="flare", cbar=False,
              xticklabels=labels, yticklabels=labels)
  plt.xlabel('Valores previstos')
  plt.ylabel('Valores reais')
  plt.title(f'Matriz de confusão {nome_modelo}');
