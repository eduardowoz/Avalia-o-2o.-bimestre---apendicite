# src/data_pipeline/normalization.py

import pandas as pd
from sklearn import preprocessing
from pickle import dump, load
from pathlib import Path

# Define caminhos para arquivos de modelos.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
NORMALIZER_MODEL_PATH = MODELS_DIR / "modelo_normalizador.pkl"

# Lista de colunas numéricas usadas no dataset.
COLUNAS_NUMERICAS = [
    'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay', 'Appendix_Diameter', 'Body_Temperature',
    'WBC_Count', 'Neutrophil_Percentage', 'RBC_Count', 'Hemoglobin', 'RDW', 'Thrombocyte_Count',
    'CRP', 'Alvarado_Score', 'Paedriatic_Appendicitis_Score'
]

# Lista de todas as features esperadas pelo modelo após pré-processamento.
ENCODED_FEATURES_NAMES = [
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

def normalizar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    """Normaliza features numéricas e aplica one-hot encoding em categóricas."""
    print("> Normalizando dados (MinMaxScaler e One-Hot Encoding)...")

    # Copia as colunas alvo antes do processamento das features.
    colunas_target = dados[['Severity', 'Diagnosis', 'Management']].copy()
    
    # Identifica colunas categóricas para one-hot encoding.
    colunas_categoricas_para_dummies = dados.drop(columns=COLUNAS_NUMERICAS + ['Severity', 'Diagnosis', 'Management'], errors='ignore').columns.tolist()

    # Seleciona colunas numéricas existentes.
    colunas_numericas_existentes_df = dados[[col for col in COLUNAS_NUMERICAS if col in dados.columns]].copy()

    # Cria, treina e salva o normalizador.
    normalizador = preprocessing.MinMaxScaler()
    modelo_normalizador = normalizador.fit(colunas_numericas_existentes_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(NORMALIZER_MODEL_PATH, "wb") as f:
        dump(modelo_normalizador, f)
    print(f"Normalizador salvo em: {NORMALIZER_MODEL_PATH}")

    # Normaliza as colunas numéricas.
    dados_num_normalizados = modelo_normalizador.transform(colunas_numericas_existentes_df)
    dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=colunas_numericas_existentes_df.columns)

    # Aplica one-hot encoding nas colunas categóricas.
    dados_cat_normalizados = pd.get_dummies(dados[colunas_categoricas_para_dummies], dtype=int)
    
    # Reseta índices para concatenação.
    dados_num_normalizados.reset_index(drop=True, inplace=True)
    dados_cat_normalizados.reset_index(drop=True, inplace=True)
    colunas_target.reset_index(drop=True, inplace=True)
    
    # Combina features normalizadas e one-hot encoded.
    dados_features_transformadas = pd.concat([dados_num_normalizados, dados_cat_normalizados], axis=1)

    # Garante que o DataFrame tenha todas as colunas esperadas pelo modelo.
    dados_features_completas = pd.DataFrame(columns=ENCODED_FEATURES_NAMES)
    dados_features_completas = pd.concat([dados_features_completas, dados_features_transformadas], ignore_index=True)
    dados_features_completas = dados_features_completas.fillna(0)

    # Adiciona as colunas alvo de volta.
    dados_normalizados_final = pd.concat([dados_features_completas, colunas_target], axis=1)

    print("> Normalização e One-Hot Encoding concluídos.")
    return dados_normalizados_final

def normalizar_paciente(paciente: pd.DataFrame) -> pd.DataFrame:
    """Normaliza uma nova instância de paciente para inferência."""
    print("> Normalizando dados do paciente para inferência...")
    try:
        with open(NORMALIZER_MODEL_PATH, "rb") as f:
            modelo_normalizador = load(f)
    except FileNotFoundError:
        print(f"ERRO: Modelo normalizador '{NORMALIZER_MODEL_PATH.name}' não encontrado! Execute o treinamento primeiro para gerar o modelo.")
        return None
    except Exception as e:
        print(f"ERRO ao carregar o normalizador: {e}")
        return None

    # Separa colunas numéricas e categóricas do paciente.
    colunas_numericas_paciente = paciente[[col for col in COLUNAS_NUMERICAS if col in paciente.columns]].copy()
    colunas_categoricas_paciente = paciente.drop(columns=COLUNAS_NUMERICAS, errors='ignore').columns.tolist()

    # Normaliza dados numéricos do paciente.
    paciente_num_normalizados = modelo_normalizador.transform(colunas_numericas_paciente)
    paciente_num_normalizados = pd.DataFrame(paciente_num_normalizados, columns=colunas_numericas_paciente.columns)

    # Aplica one-hot encoding em dados categóricos do paciente.
    paciente_cat_normalizados = pd.get_dummies(paciente[colunas_categoricas_paciente], dtype=int)
    
    # Combina os dataframes processados do paciente.
    paciente_normalizado_temp = pd.concat([paciente_num_normalizados, paciente_cat_normalizados], axis=1)

    # Garante que as colunas do paciente correspondam ao esperado pelo modelo.
    paciente_final = pd.DataFrame(columns=ENCODED_FEATURES_NAMES)
    paciente_final = pd.concat([paciente_final, paciente_normalizado_temp], ignore_index=True)
    paciente_final = paciente_final.fillna(0)

    return paciente_final[ENCODED_FEATURES_NAMES]

def desnormalizar_paciente(paciente_normalizado: pd.DataFrame) -> pd.DataFrame:
    """Desnormaliza as features numéricas de uma instância de paciente."""
    print("> Desnormalizando dados do paciente para exibição...")
    try:
        with open(NORMALIZER_MODEL_PATH, "rb") as f:
            modelo_normalizador = load(f)
    except FileNotFoundError:
        print(f"ERRO: Modelo normalizador '{NORMALIZER_MODEL_PATH.name}' não encontrado!")
        print("Execute o treinamento para gerar o modelo primeiro.")
        return None
    except Exception as e:
        print(f"ERRO ao carregar o normalizador para desnormalização: {e}")
        return None

    # Seleciona colunas numéricas para desnormalização.
    dados_numericos_para_reverter_cols = [col for col in COLUNAS_NUMERICAS if col in paciente_normalizado.columns]
    dados_numericos_para_reverter = paciente_normalizado[dados_numericos_para_reverter_cols]

    # Aplica a transformação inversa.
    dados_desnormalizados = modelo_normalizador.inverse_transform(dados_numericos_para_reverter)

    # Converte o resultado de volta para DataFrame.
    paciente_desnormalizado_df = pd.DataFrame(dados_desnormalizados, 
                                                columns=dados_numericos_para_reverter_cols, 
                                                index=paciente_normalizado.index)

    return paciente_desnormalizado_df