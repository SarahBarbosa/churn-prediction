import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_countplots(data, coluna_hue, grupos, paleta, figsize=(12, 8)):
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
