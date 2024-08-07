# Alura Challenge Dados 2ª Edição 📊

<p align="center">
  <img src="https://imgur.com/1yxAnVf.png" style="width: 100%;">
</p>

No [Alura Challenge Dados 2ª Edição](https://www.alura.com.br/challenges/dados-2?host=https://cursos.alura.com.br), o objetivo é desenvolver uma solução para uma empresa de telecomunicações que visa compreender e prever a Taxa de Evasão de Clientes (Churn Rate). O projeto segue um cronograma de quatro semanas, com cada semana correspondendo a uma etapa específica.

## 📋 Detalhes do Projeto: Redução da Taxa de Evasão de Clientes na Novexus

### :arrow_right: Semana 01: Limpeza e Análise Exploratória de Dados (ETL & EDA)

Na primeira semana, focamos na preparação dos dados e na obtenção de insights iniciais (ETL & EDA). As atividades incluíram:

- Compreensão do conteúdo do conjunto de dados;
- Identificação e tratamento de inconsistências;
- Análise do comportamento das features categóricas e numéricas em relação a target;
- Avaliação da correlação entre as variáveis.

**Notebook Correspondente:** [`S01.ipynb`](https://github.com/SarahBarbosa/churn-prediction/blob/main/S01.ipynb)

### :arrow_right: Semana 02: Feature Engineering e Construção do Modelo de Machine Learning (ML)

Na segunda semana, construímos e otimizamos o modelo de ML. Os processos incluíram:

- Lidar com o desbalanceamento dos dados da target usando três abordagens: Oversampling (SMOTE), Undersampling (Tomek Links) e Default (mantendo o desbalanceamento).
- Encoding dos dados categóricos usando CatBoost, normalização dos dados numéricos usando StandardScaler e construção do pipeline.
- Utilização do RepeatedStratifiedKFold com 3 folds para avaliar o desempenho dos modelos.
- Ajuste de hiperparâmetros usando Grid Search para cada abordagem.
- Avaliação dos modelos usando o Recall como métrica crítica.
- Visualização dos resultados usando a matriz de confusão e a curva ROC (com o valor da AUC).
- Explicação dos resultados usando SHAP (Feature Importance e Waterfall).

**Notebook Correspondente:** [`S02.ipynb`](https://github.com/SarahBarbosa/churn-prediction/blob/main/S02.ipynb)

### :arrow_right: Semana 03&04: Deploy do Modelo :dash:

Nas últimas duas semanas desenvolvemos um aplicativo utilizando o Streamlit para disponibilizar nosso modelo de previsão de churn para a Novexus. Esse aplicativo oferece duas opções de entrada de dados: a possibilidade de importar um arquivo CSV contendo os dados do cliente ou a inserção manual desses dados. Quando o usuário utiliza o aplicativo, ele recebe como retorno a probabilidade de um cliente deixar a empresa. Essa probabilidade é apresentada de forma visual através de um gráfico tipo "gauge chart". 

> Para acessar o aplicativo, clique no link a seguir: [Churn Predictor Novexus](https://churn-prediction-novexus1.streamlit.app/)

O código-fonte do aplicativo está disponível em: [`S03.py`](https://github.com/SarahBarbosa/churn-prediction/blob/main/S03.py).

### :arrow_right: Resultados dos Modelos :chart_with_upwards_trend:

Após o treinamento de 6 modelos (Regressão Logística, KNN, Gradient Boosting, Árvore de Decisão, Floresta Randômica e Support Vector Machine), observamos que a Regressão Logística se destacou nas três abordagens em relação ao Recall. Escolhemos mais dois modelos com melhor desempenho nessa métrica e realizaremos um ajuste de hiperparâmetros usando Grid Search. Os resultados foram os seguintes:

- Na estratégia de Oversampling, a Regressão Logística alcançou um Recall de 76.29%.
- Na estratégia de Oversampling e na estratégia Default, a Gradient Boosting alcançou um Recall de 74.67% e 73.62%, respectivamente.

Embora os três modelos tenham pontuações muito próximas, a estratégia de Oversampling se destacou. Portanto, utilizamos esses três melhores modelos para avaliar no conjunto de teste. Os resultados foram:

- A Regressão Logística com Oversampling teve o recall mais alto, mas a precisão foi um pouco baixa.
- O Gradient Boosting com Undersampling equilibrou razoavelmente a precisão e o recall.
- O Gradient Boosting com amostragem padrão teve a melhor precisão para a classe 1, mas o recall foi mais baixo.

Dado o setor de telecomunicações, onde o custo de atrair novos clientes é alto, minimizar a perda de clientes é fundamental. Portanto, consideramos o modelo Regressão Logística com oversampling como a escolha mais adequada para prever a probabilidade de um cliente churn.

### :arrow_right: Resumo das Recomendações 🚀

- Priorizar Contratos de Longo Prazo: A análise de dados demonstrou que clientes com contratos de maior duração têm maior probabilidade de permanecer na Novexus. Recomendamos que a empresa concentre-se em oferecer planos de contrato anual tradicionais, alocando recursos significativos de marketing e promoções para esses planos.

- Planos Sem Contrato Fixo como Alternativa: Os planos sem contrato fixo podem ser oferecidos como uma opção secundária, mantendo o foco principal nos planos de contrato anual.

- Promoção da Fibra Óptica: Para combater a alta taxa de churn entre os clientes de fibra óptica, considere oferecer descontos especiais para incentivá-los a permanecer. Comunique claramente esses benefícios aos clientes de fibra óptica.

- Redução do Processamento de Cheques Eletrônicos: Avalie a possibilidade de incentivar os clientes a migrarem para métodos de pagamento mais eficientes, como pagamentos com cartão de crédito. Ofereça incentivos para essa transição.

**Recomendações detalhadas:** Última seção no arquivo [`S02.ipynb`](https://github.com/SarahBarbosa/churn-prediction/blob/main/S02.ipynb)

> Status do Projeto: Concluido :heavy_check_mark:

