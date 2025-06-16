# src/app/inference.py

import csv
from pickle import load
from pathlib import Path
from colorama import init, Fore, Style
import pandas as pd

# Importa funções para manipulação de dados transformados
from data_pipeline.normalization import reverter_escalonamento_paciente

# Inicializa o colorama para efeitos visuais no terminal.
init()

# Define os caminhos essenciais para o sistema.
DIRETORIO_RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
DIRETORIO_MODELOS = DIRETORIO_RAIZ_PROJETO / "models"
DIRETORIO_DADOS_SALVOS = DIRETORIO_RAIZ_PROJETO / "data"
CAMINHO_REGISTRO_PACIENTES = DIRETORIO_DADOS_SALVOS / "registro_pacientes_analisados.csv"

# Lista de identificadores de features no formato pós-processamento.
LISTA_DE_FEATURES_PROCESSADAS = [
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

# Colunas originais a serem mantidas no registro final do paciente.
COLUNAS_PARA_REGISTRO_ORIGINAL = [
    'Age','BMI','Sex','Height','Weight','Length_of_Stay','Alvarado_Score','Paedriatic_Appendicitis_Score',
    'Appendix_on_US','Appendix_Diameter','Migratory_Pain','Lower_Right_Abd_Pain',
    'Contralateral_Rebound_Tenderness','Coughing_Pain','Nausea','Loss_of_Appetite','Body_Temperature',
    'WBC_Count','Neutrophil_Percentage','Neutrophilia','RBC_Count','Hemoglobin','RDW','Thrombocyte_Count',
    'Ketones_in_Urine','RBC_in_Urine','WBC_in_Urine','CRP','Dysuria','Stool','Peritonitis','Psoas_Sign',
    'Ipsilateral_Rebound_Tenderness','US_Performed','Free_Fluids'
]

# Nomes dos resultados fornecidos pelos modelos.
NOMES_DOS_RESULTADOS_PREDITOS = ['Diagnosis', 'Severity', 'Management']

# Cabeçalho completo para o arquivo de registro CSV.
CABECALHO_CSV_FINAL = COLUNAS_PARA_REGISTRO_ORIGINAL + NOMES_DOS_RESULTADOS_PREDITOS


def reverter_codificacao_categorica(dados_codificados_series: pd.Series) -> dict:
    """Reverte dados one-hot encoded de volta para o formato categórico original."""
    dados_recuperados = {}

    mapa_para_reverter = {
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

    for coluna_base in COLUNAS_PARA_REGISTRO_ORIGINAL:
        if coluna_base in mapa_para_reverter:
            info_reversao = mapa_para_reverter[coluna_base]
            valor_encontrado = False
            for valor_original_map in info_reversao['valores']:
                coluna_one_hot = f"{info_reversao['prefixo']}{valor_original_map}"
                if coluna_one_hot in dados_codificados_series.index and dados_codificados_series.get(coluna_one_hot) == 1:
                    dados_recuperados[coluna_base] = valor_original_map
                    valor_encontrado = True
                    break
            if not valor_encontrado:
                dados_recuperados[coluna_base] = 'N/A'
        else:
            if coluna_base in dados_codificados_series.index:
                dados_recuperados[coluna_base] = dados_codificados_series[coluna_base]
            else:
                dados_recuperados[coluna_base] = 'N/A'
            
    return dados_recuperados

def carregar_e_inferir_modelo(dados_paciente_processados: pd.DataFrame, alvo_predicao: str):
    """
    Carrega um modelo de aprendizado de máquina e realiza a predição de probabilidades.
    Retorna as probabilidades preditas ou None se o modelo não for acessível.
    """
    caminho_modelo_preditor = DIRETORIO_MODELOS / f"modelo_apendicite_pediatrica_{alvo_predicao.lower()}.pkl"
    try:
        with open(caminho_modelo_preditor, "rb") as f:
            modelo_preditor = load(f)
            return modelo_preditor.predict_proba(dados_paciente_processados)
    except FileNotFoundError:
        print(f"{Fore.RED}ERRO: Modelo '{caminho_modelo_preditor.name}' não encontrado! Verifique se o treinamento foi executado.{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"{Fore.RED}ERRO ao carregar ou usar o modelo '{caminho_modelo_preditor.name}': {e}{Style.RESET_ALL}")
        return None

def registrar_dados_analisados(registro_dict: dict):
    """
    Salva os dados do paciente, incluindo os resultados das análises, em um arquivo CSV.
    """
    try:
        DIRETORIO_DADOS_SALVOS.mkdir(parents=True, exist_ok=True)
        escrever_cabecalho = not CAMINHO_REGISTRO_PACIENTES.exists() or CAMINHO_REGISTRO_PACIENTES.stat().st_size == 0

        with open(CAMINHO_REGISTRO_PACIENTES, "a", newline="", encoding="utf-8") as f:
            escritor_csv = csv.DictWriter(f, fieldnames=CABECALHO_CSV_FINAL)
            if escrever_cabecalho:
                escritor_csv.writeheader()
            
            linha_para_escrita = {col: registro_dict.get(col, '') for col in CABECALHO_CSV_FINAL}
            escritor_csv.writerow(linha_para_escrita)
        print(f"\n{Fore.CYAN}--- Análise registrada em '{CAMINHO_REGISTRO_PACIENTES.name}' ---{Style.RESET_ALL}")
            
    except IOError as e:
        print(f"{Fore.RED}ERRO AO REGISTRAR ANÁLISE EM CSV: {e}{Style.RESET_ALL}")

def realizar_analise_paciente(paciente_preparado_df: pd.DataFrame):
    """
    Conduz o processo completo de análise para um paciente,
    fornecendo diagnóstico, severidade e sugestão de manejo.
    """
    representacao_original_paciente = paciente_preparado_df.iloc[0].copy()
    
    dados_para_registro = reverter_codificacao_categorica(representacao_original_paciente)
    
    dados_metricas_originais = reverter_escalonamento_paciente(paciente_preparado_df)
    if dados_metricas_originais is not None:
        for coluna_metrica in dados_metricas_originais.columns:
            dados_para_registro[coluna_metrica] = dados_metricas_originais[coluna_metrica].iloc[0]

    # Predição de Diagnóstico.
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}=========== PREDITOR DE DIAGNÓSTICO ==========={Style.RESET_ALL}")
    probabilidades_diagnostico = carregar_e_inferir_modelo(paciente_preparado_df, 'Diagnosis')
    if probabilidades_diagnostico is None: 
        dados_para_registro['Diagnosis'] = 'ERRO'
        dados_para_registro['Severity'] = 'ERRO'
        dados_para_registro['Management'] = 'ERRO'
        registrar_dados_analisados(dados_para_registro)
        return
    
    with open(DIRETORIO_MODELOS / 'modelo_apendicite_pediatrica_diagnosis.pkl', "rb") as f:
        modelo_diagnostico = load(f)
    classe_diagnostico_predita = modelo_diagnostico.predict(paciente_preparado_df)[0]
    
    lista_prob_diagnostico = probabilidades_diagnostico[0]
    classes_do_modelo_diagnostico = modelo_diagnostico.classes_
    
    prob_da_predicao = lista_prob_diagnostico[list(classes_do_modelo_diagnostico).index(classe_diagnostico_predita)]

    dados_para_registro['Diagnosis'] = classe_diagnostico_predita
    
    dados_para_registro['Severity'] = 'N/A'
    dados_para_registro['Management'] = 'N/A'

    if classe_diagnostico_predita == 'appendicitis':
        print(f"{Fore.GREEN}Resultado: {prob_da_predicao:.2%} de chance - {dados_para_registro['Diagnosis']}{Style.RESET_ALL}")

        # Predição de Severidade (somente se diagnóstico for apendicite).
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}=== PREDITOR DE SEVERIDADE ==={Style.RESET_ALL}")
        probabilidades_severidade = carregar_e_inferir_modelo(paciente_preparado_df, 'Severity')
        
        if probabilidades_severidade is not None:
            with open(DIRETORIO_MODELOS / 'modelo_apendicite_pediatrica_severity.pkl', "rb") as f:
                modelo_severidade = load(f)
            classe_severidade_predita = modelo_severidade.predict(paciente_preparado_df)[0]
            lista_prob_severidade = probabilidades_severidade[0]
            classes_do_modelo_severidade = modelo_severidade.classes_
            prob_da_predicao_severidade = lista_prob_severidade[list(classes_do_modelo_severidade).index(classe_severidade_predita)]

            dados_para_registro['Severity'] = classe_severidade_predita
            
            if classe_severidade_predita == 'complicated':
                print(f"{Fore.RED}Resultado: {prob_da_predicao_severidade:.2%} de chance - {classe_severidade_predita}{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}Resultado: {prob_da_predicao_severidade:.2%} de chance - {classe_severidade_predita}{Style.RESET_ALL}")

        # Sugestão de Manejo (somente se diagnóstico for apendicite).
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}=== SUGESTOR DE MANEJO ==={Style.RESET_ALL}")
        probabilidades_manejo = carregar_e_inferir_modelo(paciente_preparado_df, 'Management')
        
        if probabilidades_manejo is not None:
            with open(DIRETORIO_MODELOS / 'modelo_apendicite_pediatrica_management.pkl', "rb") as f:
                modelo_manejo = load(f)
            classe_manejo_predita = modelo_manejo.predict(paciente_preparado_df)[0]
            lista_prob_manejo = probabilidades_manejo[0]
            classes_do_modelo_manejo = modelo_manejo.classes_
            prob_da_predicao_manejo = lista_prob_manejo[list(classes_do_modelo_manejo).index(classe_manejo_predita)]

            dados_para_registro['Management'] = classe_manejo_predita
            
            if classe_manejo_predita == 'conservative':
                print(f"{Fore.BLUE}Resultado: {prob_da_predicao_manejo:.2%} de chance - {classe_manejo_predita}{Style.RESET_ALL}")
            else:
                print(f"{Fore.MAGENTA}Resultado: {prob_da_predicao_manejo:.2%} de chance - {classe_manejo_predita}{Style.RESET_ALL}")

    else:
        with open(DIRETORIO_MODELOS / 'modelo_apendicite_pediatrica_diagnosis.pkl', "rb") as f:
            modelo_diagnostico = load(f)
        lista_prob_diagnostico = probabilidades_diagnostico[0]
        classes_do_modelo_diagnostico = modelo_diagnostico.classes_
        
        prob_nao_apendicite = lista_prob_diagnostico[list(classes_do_modelo_diagnostico).index('no appendicitis')]
        
        print(f"{Fore.RED}Resultado: {prob_nao_apendicite:.2%} de chance - {dados_para_registro['Diagnosis']}{Style.RESET_ALL}")

    registrar_dados_analisados(dados_para_registro)