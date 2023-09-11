from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import pandas as pd

def cross_validation_models_set(abordagem, X_train, y_train, classificadores, cv=3, random_state=42):
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
    stratified_kfold = StratifiedKFold(n_splits = cv, shuffle = True, random_state = random_state)
    resultados = []

    for nome, classificador in classificadores:
      if abordagem == 'oversampling':
        roc_auc = cross_val_score(classificador, X_train, y_train, cv=cv, scoring='roc_auc').mean()
        accuracy = cross_val_score(classificador, X_train, y_train, cv=cv, scoring='accuracy').mean()
        precision = cross_val_score(classificador, X_train, y_train, cv=cv, scoring='precision').mean()
        recall = cross_val_score(classificador, X_train, y_train, cv=cv, scoring='recall').mean()
        f1 = cross_val_score(classificador, X_train, y_train, cv=cv, scoring='f1').mean()
      else:
        roc_auc = cross_val_score(classificador, X_train, y_train, cv=stratified_kfold, scoring='roc_auc').mean()
        accuracy = cross_val_score(classificador, X_train, y_train, cv=stratified_kfold, scoring='accuracy').mean()
        precision = cross_val_score(classificador, X_train, y_train, cv=stratified_kfold, scoring='precision').mean()
        recall = cross_val_score(classificador, X_train, y_train, cv=stratified_kfold, scoring='recall').mean()
        f1 = cross_val_score(classificador, X_train, y_train, cv=stratified_kfold, scoring='f1').mean()
      

      resultados.append([nome,
                        round(roc_auc.mean(), 3),
                        round(accuracy, 3),
                        round(precision, 3),
                        round(recall, 3),
                        round(f1, 3)])
      
    df_resultados = pd.DataFrame(resultados, columns=['Modelo', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    df_style = df_resultados.style.background_gradient(cmap='Spectral_r', subset=['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'], axis=0)
      
    return df_resultados, df_style

def hyperparameter_optimization(X_train, y_train, modelos, cv=3, scoring=['accuracy', 'f1_macro']):
    """
    Otimiza hiperparâmetros para modelos de classificação usando GridSearchCV.

    Parâmetros:
    X_train (array-like): Conjunto de treinamento das features.
    y_train (array-like): Conjunto de treinamento das targets.
    modelos (list): Uma lista de tuplas contendo o nome do modelo, o modelo e os parâmetros a serem otimizados.
    cv (int): Número de dobras (folds) para validação cruzada.
    scoring (list): Lista de métricas de avaliação a serem usadas para otimização.

    Retorno:
    resultados (DataFrame): Um DataFrame contendo os melhores resultados de hiperparâmetros 
    para cada modelo.
    """
    melhor_score = []

    for nome_modelo, modelo, parametros in modelos:
        grid_search = GridSearchCV(modelo, parametros, cv=cv, scoring=scoring, refit='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"# Melhores parâmetros para {nome_modelo}: {grid_search.best_params_}")

        best_accuracy = grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]
        best_f1_macro = grid_search.best_score_

        score = {
            'Modelo': nome_modelo,
            'F1 Macro Score': best_f1_macro,
            'Accuracy': best_accuracy
        }
        melhor_score.append(score)

    return pd.DataFrame(melhor_score)