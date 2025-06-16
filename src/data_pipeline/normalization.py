# src/data_pipeline/normalization.py

import pandas as pd
from sklearn import preprocessing
from pickle import dump, load
from pathlib import Path

# Define caminhos para arquivos de artefatos do modelo.
DIRETORIO_RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
DIRETORIO_MODELOS = DIRETORIO_RAIZ_PROJETO / "models"
CAMINHO_MODELO_ESCALONADOR = DIRETORIO_MODELOS / "modelo_escalonador_minmax.pkl"

# Conjunto de identificadores para colunas numéricas no dataset.
CONJUNTO_DE_METRICAS_QUANTITATIVAS = [
    'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay', 'Appendix_Diameter', 'Body_Temperature',
    'WBC_Count', 'Neutrophil_Percentage', 'RBC_Count', 'Hemoglobin', 'RDW', 'Thrombocyte_Count',
    'CRP', 'Alvarado_Score', 'Paedriatic_Appendicitis_Score'
]

# Lista exata de todas as features esperadas pelo modelo de ML, incluindo as colunas geradas pelo one-hot encoding.
# É crucial que esta lista seja consistente entre o treinamento e a inferência.
NOME_DAS_FEATURES_CODIFICADAS_ESPERADAS = [
    'Age','BMI','Height','Weight','Length_of_Stay','Appendix_Diameter','Body_Temperature','WBC_Count',
    'Neutrophil_Percentage','RBC_Count','Hemoglobin','RDW','Thrombocyte_Count','CRP','Alvarado_Score',
    'Paedriatic_Appendicitis_Score','Sex_female','Sex_male','Appendix_on_US_no','Appendix_on_US_yes',
    'Migratory_Pain_no','Migratory_Pain_yes','Lower_Right_Abd_Pain_no','Lower_Right_Abd_Pain_yes',
    'Contralateral_Rebound_Tenderness_no','Contralateral_Rebound_Tenderness_yes','Coughing_Pain_no',
    'Coughing_Pain_yes','Nausea_no','Nausea_yes','Loss_of_Appetite_no','Loss_of_Appetite_yes',
    'Neutrophilia_no','Neutrophilia_yes','Ketones_in_Urine_+','Ketones_in_Urine_++',
    'Ketones_in_Urine_+++','Ketones_in_Urine_no','RBC_in_Urine_+','RBC_in_Urine_++',
    'RBC_in_Urine_+++','RBC_in_Urine_no','WBC_in_Urine_+','WBC_in_Urine_++',
    'WBC_in_Urine_+++','WBC_in_Urine_no','Dysuria_no','Dysuria_yes','Stool_constipation',
    'Stool_constipation, diarrhea','Stool_diarrhea','Stool_normal','Peritonitis_generalized',
    'Peritonitis_local','Peritonitis_no','Psoas_Sign_no','Psoas_Sign_yes',
    'Ipsilateral_Rebound_Tenderness_no','Ipsilateral_Rebound_Tenderness_yes','US_Performed_no',
    'US_Performed_yes','Free_Fluids_no','Free_Fluids_yes'
]

def aplicar_escalonamento_e_codificacao(estrutura_dados: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma as features do DataFrame:
    - Escalonamento de métricas quantitativas (MinMaxScaler).
    - Codificação one-hot de variáveis categóricas.
    O escalonador treinado é salvo para uso futuro na inferência.
    """
    print("> Escalonando métricas quantitativas e codificando variáveis categóricas...")

    nomes_colunas_alvo = ['Severity', 'Diagnosis', 'Management']
    valores_alvo_df = estrutura_dados[nomes_colunas_alvo].copy()
    
    categorias_para_codificar = estrutura_dados.drop(columns=CONJUNTO_DE_METRICAS_QUANTITATIVAS + nomes_colunas_alvo, errors='ignore').columns.tolist()

    metricas_quantitativas_presentes = estrutura_dados[[col for col in CONJUNTO_DE_METRICAS_QUANTITATIVAS if col in estrutura_dados.columns]].copy()

    escalonador = preprocessing.MinMaxScaler()
    instancia_escalonador_ajustada = escalonador.fit(metricas_quantitativas_presentes)

    DIRETORIO_MODELOS.mkdir(parents=True, exist_ok=True)
    
    with open(CAMINHO_MODELO_ESCALONADOR, "wb") as f:
        dump(instancia_escalonador_ajustada, f)
    print(f"Escalonador de métricas salvo em: {CAMINHO_MODELO_ESCALONADOR}")

    metricas_escalonadas = instancia_escalonador_ajustada.transform(metricas_quantitativas_presentes)
    metricas_escalonadas_df = pd.DataFrame(metricas_escalonadas, columns=metricas_quantitativas_presentes.columns)

    categorias_codificadas_df = pd.get_dummies(estrutura_dados[categorias_para_codificar], dtype=int)
    
    metricas_escalonadas_df.reset_index(drop=True, inplace=True)
    categorias_codificadas_df.reset_index(drop=True, inplace=True)
    valores_alvo_df.reset_index(drop=True, inplace=True)
    
    features_transformadas_completas = pd.concat([metricas_escalonadas_df, categorias_codificadas_df], axis=1)

    features_para_modelo = pd.DataFrame(columns=NOME_DAS_FEATURES_CODIFICADAS_ESPERADAS)
    features_para_modelo = pd.concat([features_para_modelo, features_transformadas_completas], ignore_index=True)
    features_para_modelo = features_para_modelo.fillna(0)

    conjunto_dados_transformado = pd.concat([features_para_modelo, valores_alvo_df], axis=1)

    print("> Escalonamento e codificação concluídos.")
    return conjunto_dados_transformado

def preparar_paciente_para_inferencia(dados_paciente_brutos: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara uma nova instância de paciente para inferência.
    Aplica o mesmo escalonamento e codificação one-hot usados no treinamento.
    """
    print("> Preparando dados do paciente para inferência...")
    try:
        with open(CAMINHO_MODELO_ESCALONADOR, "rb") as f:
            instancia_escalonador_carregada = load(f)
    except FileNotFoundError:
        print(f"ERRO: Escalonador de métricas '{CAMINHO_MODELO_ESCALONADOR.name}' não encontrado! Execute o treinamento primeiro para gerá-lo.")
        return None
    except Exception as e:
        print(f"ERRO ao carregar o escalonador para preparação do paciente: {e}")
        return None

    metricas_quantitativas_paciente = dados_paciente_brutos[[col for col in CONJUNTO_DE_METRICAS_QUANTITATIVAS if col in dados_paciente_brutos.columns]].copy()
    categorias_paciente = dados_paciente_brutos.drop(columns=CONJUNTO_DE_METRICAS_QUANTITATIVAS, errors='ignore').columns.tolist()

    metricas_paciente_escalonadas = instancia_escalonador_carregada.transform(metricas_quantitativas_paciente)
    metricas_paciente_escalonadas_df = pd.DataFrame(metricas_paciente_escalonadas, columns=metricas_quantitativas_paciente.columns)

    categorias_paciente_codificadas_df = pd.get_dummies(dados_paciente_brutos[categorias_paciente], dtype=int)
    
    paciente_transformado_temp = pd.concat([metricas_paciente_escalonadas_df, categorias_paciente_codificadas_df], axis=1)

    paciente_pronto_para_modelo = pd.DataFrame(columns=NOME_DAS_FEATURES_CODIFICADAS_ESPERADAS)
    paciente_pronto_para_modelo = pd.concat([paciente_pronto_para_modelo, paciente_transformado_temp], ignore_index=True)
    paciente_pronto_para_modelo = paciente_pronto_para_modelo.fillna(0)

    return paciente_pronto_para_modelo[NOME_DAS_FEATURES_CODIFICADAS_ESPERADAS]

def reverter_escalonamento_paciente(dados_paciente_escalonados: pd.DataFrame) -> pd.DataFrame:
    """
    Reverte o escalonamento das métricas quantitativas de uma instância de paciente.
    Útil para exibir dados em seu formato original.
    """
    print("> Revertendo escalonamento de dados do paciente para exibição...")
    try:
        with open(CAMINHO_MODELO_ESCALONADOR, "rb") as f:
            instancia_escalonador_carregada = load(f)
    except FileNotFoundError:
        print(f"ERRO: Escalonador de métricas '{CAMINHO_MODELO_ESCALONADOR.name}' não encontrado! Verifique se o treinamento foi executado.")
        return None
    except Exception as e:
        print(f"ERRO ao carregar o escalonador para reversão: {e}")
        return None

    metricas_quantitativas_para_reverter = [col for col in CONJUNTO_DE_METRICAS_QUANTITATIVAS if col in dados_paciente_escalonados.columns]
    dados_para_reverter = dados_paciente_escalonados[metricas_quantitativas_para_reverter]

    dados_desescalonados = instancia_escalonador_carregada.inverse_transform(dados_para_reverter)

    paciente_com_metricas_originais = pd.DataFrame(dados_desescalonados, 
                                                columns=metricas_quantitativas_para_reverter, 
                                                index=dados_paciente_escalonados.index)

    return paciente_com_metricas_originais