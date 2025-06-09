# src/training_pipeline/train.py

import pandas as pd
from pathlib import Path
from pickle import dump, load

# Importa funções para processamento de dados
from data_pipeline.processing import importar_dataset, preprocessar_dados
from data_pipeline.normalization import normalizar_dados, ENCODED_FEATURES_NAMES
from data_pipeline.balancing import balancear

# Importa ferramentas para treinamento de modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define o caminho para a pasta onde os modelos serão salvos
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def treinar_modelo_individual(dados: pd.DataFrame, target: str):
    """
    Treina e avalia um modelo RandomForest para um target específico.
    Salva o modelo treinado.
    """
    print(f"\n> Treinando modelo para a coluna target: {target}")

    model_path = MODELS_DIR / f'pediactric_appendicitis_{target}_model.pkl'

    # Verifica se o modelo já existe e pula o treinamento se encontrado.
    if model_path.exists():
        print(f"Modelo para {target} já existe em {model_path}. Pulando treinamento.")
        return

    # Separa as features da coluna target.
    colunas_target = ['Diagnosis', 'Severity', 'Management']
    features_cols = [col for col in dados.columns if col not in colunas_target]
    
    dados_atributos = dados[features_cols].copy()
    dados_classes = dados[target].copy()

    # Configura o modelo base.
    tree = RandomForestClassifier(random_state=42)

    # Define os hiperparâmetros para otimização do modelo.
    tree_grid = {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', None]
    }

    # Realiza a busca em grade para encontrar os melhores hiperparâmetros.
    print(f"> Realizando otimização de hiperparâmetros para {target}...")
    tree_hyperparameters = GridSearchCV(
        estimator=tree,
        param_grid=tree_grid,
        cv=5,
        verbose=1,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    tree_hyperparameters.fit(dados_atributos, dados_classes)
    
    print(f"\nMelhores parâmetros para {target}:")
    print(tree_hyperparameters.best_params_)

    # Treina o modelo final com os melhores parâmetros em todo o dataset.
    print(f"> Treinando o modelo final para {target} com os melhores parâmetros...")
    model_final = RandomForestClassifier(**tree_hyperparameters.best_params_, random_state=42)
    model_final.fit(dados_atributos, dados_classes)

    # Avalia o modelo usando validação cruzada.
    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    print(f"> Avaliando o modelo de {target} com validação cruzada (10 folds)...")
    scores_cross = cross_validate(model_final, dados_atributos, dados_classes, cv=10, scoring=scoring_metrics, n_jobs=-1)

    print(f"\n--- Resultados da validação cruzada (Target: {target}) ---")
    print(f"  Acurácia média:   {scores_cross['test_accuracy'].mean():.2%}")
    print(f"  Precisão média:   {scores_cross['test_precision_macro'].mean():.2%}")
    print(f"  Recall médio:     {scores_cross['test_recall_macro'].mean():.2%}")
    print(f"  F1-score médio:   {scores_cross['test_f1_macro'].mean():.2%}")
    print("-" * 40)

    # Salva o modelo treinado em um arquivo.
    dump(model_final, open(model_path, "wb"))
    print(f"> Modelo para {target} salvo em {model_path}")


def executar_pipeline_de_treinamento():
    """
    Executa o pipeline completo de treinamento de modelos.
    Inclui importação, pré-processamento, normalização, balanceamento
    e treinamento de modelos para cada target.
    """
    print("==================================================")
    print(">>> Iniciando Pipeline de Treinamento de Modelos <<<")
    print("==================================================")
    
    # Importa o dataset.
    dados_raw = importar_dataset()
    if dados_raw is None:
        print("Pipeline de treinamento interrompido devido a erro na importação do dataset.")
        return
#==============================================================================================================================    
    # --- INÍCIO DA DEPURAÇÃO ---
    print("\n--- INFORMAÇÕES DO DATAFRAME APÓS IMPORTAÇÃO ---")
    print("Colunas presentes no dados_raw:", dados_raw.columns.tolist())
    print("Primeiras 5 linhas do dados_raw:\n", dados_raw.head())
    print("Verificando nulos nas colunas alvo no dados_raw:\n", dados_raw[['Severity', 'Diagnosis', 'Management']].isnull().sum())
    print("--- FIM DA DEPURAÇÃO ---")
    # --- FIM DA DEPURAÇÃO ---

    # 2. Pré-processar os dados
    dados_processados = preprocessar_dados(dados_raw.copy())

    # Pré-processa os dados brutos.
    dados_processados = preprocessar_dados(dados_raw.copy())

    # Normaliza os dados, aplicando escalonamento e one-hot encoding.
    dados_normalizados = normalizar_dados(dados_processados.copy())
    print("> Dados processados e normalizados com sucesso.")

    # Define as colunas alvo para treinamento de modelos individuais.
    targets_para_treinar = ['Diagnosis', 'Severity', 'Management']

    # Treina um modelo para cada coluna alvo.
    for target in targets_para_treinar:
        df_para_treinamento = dados_normalizados.copy()

        # Filtra os dados para 'Severity' e 'Management' se o diagnóstico for 'appendicitis'.
        if target in ['Severity', 'Management']:
            df_para_treinamento = df_para_treinamento[df_para_treinamento['Diagnosis'] == 'appendicitis'].copy()
            if df_para_treinamento.empty:
                print(f"Não há dados de 'appendicitis' para treinar o modelo de {target}. Pulando.")
                continue
            
        # Balanceia as classes para o target atual.
        df_balanceado = balancear(df_para_treinamento, target)

        # Treina o modelo individual para o target.
        treinar_modelo_individual(df_balanceado, target)
    
    print("\n==================================================")
    print(">>> Pipeline de treinamento concluído com sucesso! <<<")
    print("==================================================")