# src/training_pipeline/train.py

import pandas as pd
from pathlib import Path
from pickle import dump, load

# Importa módulos de preparação e tratamento de dados
from data_pipeline.processing import obter_conjunto_de_dados_brutos, preparar_dados_para_analise
from data_pipeline.normalization import aplicar_escalonamento_e_codificacao, NOME_DAS_FEATURES_CODIFICADAS_ESPERADAS
from data_pipeline.balancing import ajustar_desbalanceamento_classes

# Importa algoritmos e ferramentas de avaliação para modelos de aprendizado de máquina
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define o diretório para armazenar os modelos treinados
DIRETORIO_RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
DIRETORIO_MODELOS = DIRETORIO_RAIZ_PROJETO / "models"
DIRETORIO_MODELOS.mkdir(exist_ok=True, parents=True)

def treinar_e_persistir_modelo(dados_para_treino: pd.DataFrame, identificador_alvo: str):
    """
    Treina um modelo RandomForest para um alvo específico, otimiza hiperparâmetros
    e avalia seu desempenho. O modelo treinado é salvo em disco.
    """
    print(f"\n> Iniciando treinamento para o alvo: {identificador_alvo}")

    caminho_do_modelo = DIRETORIO_MODELOS / f'modelo_apendicite_pediatrica_{identificador_alvo.lower()}.pkl'

    if caminho_do_modelo.exists():
        print(f"Modelo para {identificador_alvo} já existe em {caminho_do_modelo}. Treinamento ignorado.")
        return

    colunas_dos_alvos = ['Diagnosis', 'Severity', 'Management']
    colunas_dos_atributos = [col for col in dados_para_treino.columns if col not in colunas_dos_alvos]
    
    conjunto_atributos = dados_para_treino[colunas_dos_atributos].copy()
    conjunto_classes = dados_para_treino[identificador_alvo].copy()

    estimador_base = RandomForestClassifier(random_state=42)

    grade_de_parametros = {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', None]
    }

    print(f"> Otimizando hiperparâmetros para {identificador_alvo} com busca em grade...")
    busca_otimizada_parametros = GridSearchCV(
        estimator=estimador_base,
        param_grid=grade_de_parametros,
        cv=5,
        verbose=1,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    busca_otimizada_parametros.fit(conjunto_atributos, conjunto_classes)
    
    print(f"\nMelhores parâmetros para {identificador_alvo}:")
    print(busca_otimizada_parametros.best_params_)

    print(f"> Treinando o modelo final para {identificador_alvo} com parâmetros otimizados...")
    modelo_final_treinado = RandomForestClassifier(**busca_otimizada_parametros.best_params_, random_state=42)
    modelo_final_treinado.fit(conjunto_atributos, conjunto_classes)

    metricas_para_avaliacao = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    print(f"> Avaliando o desempenho do modelo de {identificador_alvo} com validação cruzada (10 folds)...")
    resultados_validacao_cruzada = cross_validate(modelo_final_treinado, conjunto_atributos, conjunto_classes, cv=10, scoring=metricas_para_avaliacao, n_jobs=-1)

    print(f"\n--- Resultados da Validação Cruzada (Alvo: {identificador_alvo}) ---")
    print(f"  Acurácia média:   {resultados_validacao_cruzada['test_accuracy'].mean():.2%}")
    print(f"  Precisão média:   {resultados_validacao_cruzada['test_precision_macro'].mean():.2%}")
    print(f"  Recall médio:     {resultados_validacao_cruzada['test_recall_macro'].mean():.2%}")
    print(f"  F1-score médio:   {resultados_validacao_cruzada['test_f1_macro'].mean():.2%}")
    print("-" * 40)

    dump(modelo_final_treinado, open(caminho_do_modelo, "wb"))
    print(f"> Modelo para {identificador_alvo} salvo em {caminho_do_modelo}")


def iniciar_pipeline_de_treinamento():
    """
    Orquestra o pipeline completo de preparação de dados e treinamento de modelos.
    As etapas incluem aquisição, pré-processamento, normalização, balanceamento
    e persistência dos modelos treinados para os alvos Diagnosis, Severity e Management.
    """
    print("==================================================")
    print(">>> Iniciando Pipeline de Geração de Modelos <<<")
    print("==================================================")
    
    dados_iniciais = obter_conjunto_de_dados_brutos()
    if dados_iniciais is None:
        print("Processo de treinamento interrompido devido a falha na aquisição dos dados.")
        return

    dados_preparados = preparar_dados_para_analise(dados_iniciais.copy())

    dados_transformados = aplicar_escalonamento_e_codificacao(dados_preparados.copy())
    print("> Dados preparados e transformados com sucesso.")

    alvos_para_modelagem = ['Diagnosis', 'Severity', 'Management']

    for alvo_atual in alvos_para_modelagem:
        conjunto_para_modelagem = dados_transformados.copy()

        if alvo_atual in ['Severity', 'Management']:
            conjunto_para_modelagem = conjunto_para_modelagem[conjunto_para_modelagem['Diagnosis'] == 'appendicitis'].copy()
            if conjunto_para_modelagem.empty:
                print(f"Não há registros de 'appendicitis' para treinar o modelo de {alvo_atual}. Treinamento ignorado.")
                continue
            
        conjunto_balanceado = ajustar_desbalanceamento_classes(conjunto_para_modelagem, alvo_atual)

        treinar_e_persistir_modelo(conjunto_balanceado, alvo_atual)
    
    print("\n==================================================")
    print(">>> Pipeline de Geração de Modelos Concluído! <<<")
    print("==================================================")