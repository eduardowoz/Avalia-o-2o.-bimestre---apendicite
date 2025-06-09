# src/app/inference.py

import csv
from pickle import load
from pathlib import Path
from colorama import init, Fore, Style
import pandas as pd

# Importa funções para desnormalização de dados
from data_pipeline.normalization import desnormalizar_paciente

# Inicializa o colorama para formatação de texto no terminal.
init()

# Define caminhos para diretórios e arquivos.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "pacientes_inferidos.csv"

# Lista de features esperadas pelo modelo após pré-processamento.
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

# Lista de colunas originais a serem mantidas no arquivo CSV de saída.
ORIGINAL_COLUMNS_TO_KEEP = [
    'Age','BMI','Sex','Height','Weight','Length_of_Stay','Alvarado_Score','Paedriatic_Appendicitis_Score',
    'Appendix_on_US','Appendix_Diameter','Migratory_Pain','Lower_Right_Abd_Pain',
    'Contralateral_Rebound_Tenderness','Coughing_Pain','Nausea','Loss_of_Appetite','Body_Temperature',
    'WBC_Count','Neutrophil_Percentage','Neutrophilia','RBC_Count','Hemoglobin','RDW','Thrombocyte_Count',
    'Ketones_in_Urine','RBC_in_Urine','WBC_in_Urine','CRP','Dysuria','Stool','Peritonitis','Psoas_Sign',
    'Ipsilateral_Rebound_Tenderness','US_Performed','Free_Fluids'
]

# Nomes das colunas de resultado da inferência.
RESULT_NAMES = ['Diagnosis', 'Severity', 'Management']

# Cabeçalho completo para o arquivo CSV de saída.
FINAL_CSV_HEADERS = ORIGINAL_COLUMNS_TO_KEEP + RESULT_NAMES


def reverter_one_hot_encoding(paciente_encoded_data_series: pd.Series) -> dict:
    """Reverte dados one-hot encoded para o formato original."""
    dados_revertidos = {}

    # Mapeamento para reverter colunas específicas.
    mapa_reversao = {
        'Sex': {'prefixo': 'Sex_', 'valores': ['female', 'male']},
        'Appendix_on_US': {'prefixo': 'Appendix_on_US_', 'valores': ['no', 'yes']},
        'Migratory_Pain': {'prefixo': 'Migratory_Pain_', 'valores': ['no', 'yes']},
        'Lower_Right_Abd_Pain': {'prefixo': 'Lower_Right_Abd_Pain_', 'valores': ['no', 'yes']},
        'Contralateral_Rebound_Tenderness': {'prefixo': 'Contralateral_Rebound_Tenderness_', 'valores': ['no', 'yes']},
        'Coughing_Pain': {'prefixo': 'Coughing_Pain_', 'valores': ['no', 'yes']},
        'Nausea': {'prefixo': 'Nausea_', 'valores': ['no', 'yes']},
        'Loss_of_Appetite': {'prefixo': 'Loss_of_Appetite_', 'valores': ['no', 'yes']},
        'Neutrophilia': {'prefixo': 'Neutrophilia_', 'valores': ['no', 'yes']},
        'Ketones_in_Urine': {'prefixo': 'Ketones_in_Urine_', 'valores': ['+', '++', '+++', 'no']},
        'RBC_in_Urine': {'prefixo': 'RBC_in_Urine_', 'valores': ['+', '++', '+++', 'no']},
        'WBC_in_Urine': {'prefixo': 'WBC_in_Urine_', 'valores': ['+', '++', '+++', 'no']},
        'Dysuria': {'prefixo': 'Dysuria_', 'valores': ['no', 'yes']},
        'Stool': {'prefixo': 'Stool_', 'valores': ['constipation', 'constipation, diarrhea', 'diarrhea', 'normal']},
        'Peritonitis': {'prefixo': 'Peritonitis_', 'valores': ['generalized', 'local', 'no']},
        'Psoas_Sign': {'prefixo': 'Psoas_Sign_', 'valores': ['no', 'yes']},
        'Ipsilateral_Rebound_Tenderness': {'prefixo': 'Ipsilateral_Rebound_Tenderness_', 'valores': ['no', 'yes']},
        'US_Performed': {'prefixo': 'US_Performed_', 'valores': ['no', 'yes']},
        'Free_Fluids': {'prefixo': 'Free_Fluids_', 'valores': ['no', 'yes']},
    }

    # Processa cada coluna para reverter o encoding.
    for col_original in ORIGINAL_COLUMNS_TO_KEEP:
        if col_original in mapa_reversao:
            info = mapa_reversao[col_original]
            found_value = False
            for valor_original in info['valores']:
                coluna_codificada = f"{info['prefixo']}{valor_original}"
                if coluna_codificada in paciente_encoded_data_series.index and paciente_encoded_data_series.get(coluna_codificada) == 1:
                    dados_revertidos[col_original] = valor_original
                    found_value = True
                    break
            if not found_value:
                dados_revertidos[col_original] = 'N/A'
        else:
            if col_original in paciente_encoded_data_series.index:
                dados_revertidos[col_original] = paciente_encoded_data_series[col_original]
            else:
                dados_revertidos[col_original] = 'N/A'
            
    return dados_revertidos

def inferir_target(paciente_df: pd.DataFrame, target: str):
    """
    Carrega um modelo e faz a predição de probabilidade para um target.
    Retorna as probabilidades ou None em caso de erro.
    """
    model_path = MODELS_DIR / f"pediactric_appendicitis_{target}_model.pkl"
    try:
        with open(model_path, "rb") as f:
            model = load(f)
            return model.predict_proba(paciente_df)
    except FileNotFoundError:
        print(f"{Fore.RED}ERRO: Modelo '{model_path.name}' não encontrado! Execute o treinamento primeiro.{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"{Fore.RED}ERRO ao carregar ou usar o modelo '{model_path.name}': {e}{Style.RESET_ALL}")
        return None

def salvar_inferencia_csv(dados_dict: dict):
    """Salva os dados do paciente e resultados da inferência em um arquivo CSV."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        escrever_cabecalho = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_CSV_HEADERS)
            if escrever_cabecalho:
                writer.writeheader()
            
            row_to_write = {col: dados_dict.get(col, '') for col in FINAL_CSV_HEADERS}
            writer.writerow(row_to_write)
        print(f"\n{Fore.CYAN}--- Inferência salva em '{CSV_PATH.name}' ---{Style.RESET_ALL}")
            
    except IOError as e:
        print(f"{Fore.RED}ERRO AO SALVAR CSV: {e}{Style.RESET_ALL}")

def inferir_paciente(paciente_normalizado_df: pd.DataFrame):
    """
    Executa o processo de inferência completo para um paciente.
    Realiza previsões para diagnóstico, gravidade e tratamento.
    """
    # Desnormaliza dados numéricos e reverte one-hot encoding para exibição/salvamento.
    paciente_original_representation = paciente_normalizado_df.iloc[0].copy()
    
    dados_para_salvar = reverter_one_hot_encoding(paciente_original_representation)
    
    dados_desnormalizados_num = desnormalizar_paciente(paciente_normalizado_df)
    if dados_desnormalizados_num is not None:
        for col in dados_desnormalizados_num.columns:
            dados_para_salvar[col] = dados_desnormalizados_num[col].iloc[0]

    # Processa o diagnóstico.
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}=========== DIAGNÓSTICO DE APENDICITE ==========={Style.RESET_ALL}")
    diagnostico_proba = inferir_target(paciente_normalizado_df, 'Diagnosis')
    if diagnostico_proba is None: 
        dados_para_salvar['Diagnosis'] = 'ERRO'
        dados_para_salvar['Severity'] = 'ERRO'
        dados_para_salvar['Management'] = 'ERRO'
        salvar_inferencia_csv(dados_para_salvar)
        return
    
    with open(MODELS_DIR / 'pediactric_appendicitis_Diagnosis_model.pkl', "rb") as f:
        model_diagnosis = load(f)
    diag_pred_class = model_diagnosis.predict(paciente_normalizado_df)[0]
    
    diag_proba_list = diagnostico_proba[0]
    diag_classes = model_diagnosis.classes_
    
    prob_predita = diag_proba_list[list(diag_classes).index(diag_pred_class)]

    dados_para_salvar['Diagnosis'] = diag_pred_class
    
    dados_para_salvar['Severity'] = 'N/A'
    dados_para_salvar['Management'] = 'N/A'

    if diag_pred_class == 'appendicitis':
        print(f"{Fore.GREEN}Resultado: {prob_predita:.2%} de chance - {dados_para_salvar['Diagnosis']}{Style.RESET_ALL}")

        # Processa a gravidade, se houver apendicite.
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}=== GRAVIDADE DA APENDICITE ==={Style.RESET_ALL}")
        severity_proba = inferir_target(paciente_normalizado_df, 'Severity')
        
        if severity_proba is not None:
            with open(MODELS_DIR / 'pediactric_appendicitis_Severity_model.pkl', "rb") as f:
                model_severity = load(f)
            sev_pred_class = model_severity.predict(paciente_normalizado_df)[0]
            sev_proba_list = severity_proba[0]
            sev_classes = model_severity.classes_
            prob_sev_predita = sev_proba_list[list(sev_classes).index(sev_pred_class)]

            dados_para_salvar['Severity'] = sev_pred_class
            
            if sev_pred_class == 'complicated':
                print(f"{Fore.RED}Resultado: {prob_sev_predita:.2%} de chance - {sev_pred_class}{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}Resultado: {prob_sev_predita:.2%} de chance - {sev_pred_class}{Style.RESET_ALL}")

        # Processa o tratamento, se houver apendicite.
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}=== TRATAMENTO RECOMENDADO ==={Style.RESET_ALL}")
        management_proba = inferir_target(paciente_normalizado_df, 'Management')
        
        if management_proba is not None:
            with open(MODELS_DIR / 'pediactric_appendicitis_Management_model.pkl', "rb") as f:
                model_management = load(f)
            mgmt_pred_class = model_management.predict(paciente_normalizado_df)[0]
            mgmt_proba_list = management_proba[0]
            mgmt_classes = model_management.classes_
            prob_mgmt_predita = mgmt_proba_list[list(mgmt_classes).index(mgmt_pred_class)]

            dados_para_salvar['Management'] = mgmt_pred_class
            
            if mgmt_pred_class == 'conservative':
                print(f"{Fore.BLUE}Resultado: {prob_mgmt_predita:.2%} de chance - {mgmt_pred_class}{Style.RESET_ALL}")
            else:
                print(f"{Fore.MAGENTA}Resultado: {prob_mgmt_predita:.2%} de chance - {mgmt_pred_class}{Style.RESET_ALL}")

    else: # Se o diagnóstico não for apendicite.
        model_diagnosis = load(MODELS_DIR / 'pediactric_appendicitis_Diagnosis_model.pkl')
        diag_proba_list = diagnostico_proba[0]
        diag_classes = model_diagnosis.classes_
        
        prob_no_appendicitis = diag_proba_list[list(diag_classes).index('no appendicitis')]
        
        print(f"{Fore.RED}Resultado: {prob_no_appendicitis:.2%} de chance - {dados_para_salvar['Diagnosis']}{Style.RESET_ALL}")

    salvar_inferencia_csv(dados_para_salvar)