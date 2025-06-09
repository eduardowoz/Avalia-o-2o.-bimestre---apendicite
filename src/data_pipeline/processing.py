# src/data_pipeline/processing.py

import pandas as pd
from ucimlrepo import fetch_ucirepo

def importar_dataset():
    """Importa o dataset de apendicite pediátrica."""
    print("> Buscando dataset do repositório UCI...")

    try:
        df = pd.read_csv('data/app_data.csv', sep=';')
        print("> Dataset 'app_data.csv' importado com sucesso da pasta 'data/'.")
    except FileNotFoundError:
        print("ERRO: O arquivo 'app_data.csv' não foi encontrado na pasta 'data/'.")
        print("Certifique-se de que a base de dados foi movida para o local correto.")
        return None
    except Exception as e:
        print(f"ERRO ao carregar o dataset: {e}")
        return None

    return df

def preencher_moda(dados, coluna):
    """Preenche valores faltantes de uma coluna com a moda."""
    if coluna in dados.columns and not dados[coluna].isnull().all():
        moda = dados[coluna].mode()[0]
        dados[coluna] = dados[coluna].fillna(moda)
    elif coluna not in dados.columns:
        print(f"Aviso: Coluna '{coluna}' não encontrada no DataFrame para preenchimento por moda.")
    return dados

def preencher_mediana(dados, coluna):
    """Preenche valores faltantes de uma coluna com a mediana."""
    if coluna in dados.columns and pd.api.types.is_numeric_dtype(dados[coluna]) and not dados[coluna].isnull().all():
        mediana = dados[coluna].median()
        dados[coluna] = dados[coluna].fillna(mediana)
    elif coluna not in dados.columns:
        print(f"Aviso: Coluna '{coluna}' não encontrada no DataFrame para preenchimento por mediana.")
    elif not pd.api.types.is_numeric_dtype(dados[coluna]):
        print(f"Aviso: Coluna '{coluna}' não é numérica. Imputação por mediana ignorada.")
    return dados

def preprocessar_dados(dados):
    """Realiza o pré-processamento dos dados, removendo colunas e preenchendo faltantes."""
    print("> Iniciando pré-processamento dos dados...")

    # Colunas para remover do dataset.
    colunas_para_remover = [
        'Segmented_Neutrophils', 'Appendix_Wall_Layers', 'Target_Sign', 'Appendicolith',
        'Perfusion', 'Perforation', 'Surrounding_Tissue_Reaction', 'Appendicular_Abscess',
        'Abscess_Location', 'Pathological_Lymph_Nodes', 'Lymph_Nodes_Location',
        'Bowel_Wall_Thickening', 'Conglomerate_of_Bowel_Loops', 'Ileus', 'Coprostasis',
        'Meteorism', 'Enteritis', 'Gynecological_Findings'
    ]

    colunas_para_remover_existentes = [col for col in colunas_para_remover if col in dados.columns]
    dados = dados.drop(columns=colunas_para_remover_existentes, errors='ignore')
    print(f"Colunas removidas: {colunas_para_remover_existentes}")

    # Colunas categóricas que terão valores faltantes preenchidos por moda.
    colunas_categoricas_moda = [
        'Sex', 'Neutrophilia', 'Ketones_in_Urine', 'Stool',
        'Contralateral_Rebound_Tenderness', 'Coughing_Pain', 'Nausea', 'Loss_of_Appetite',
        'RBC_in_Urine', 'WBC_in_Urine', 'Dysuria', 'Peritonitis', 'Psoas_Sign',
        'Ipsilateral_Rebound_Tenderness', 'US_Performed', 'Free_Fluids',
        'Appendix_on_US', 'Migratory_Pain', 'Lower_Right_Abd_Pain',
        'Diagnosis', 'Severity', 'Management' # Colunas alvo também são preenchidas.
    ]

    # Colunas numéricas que terão valores faltantes preenchidos por mediana.
    colunas_numericas_mediana = [
        'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay', 'Appendix_Diameter',
        'Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage', 'RBC_Count',
        'Hemoglobin', 'RDW', 'Thrombocyte_Count', 'CRP'
    ]

    # Colunas de score que terão valores faltantes preenchidos por moda.
    colunas_score_moda = ['Alvarado_Score', 'Paedriatic_Appendicitis_Score']

    # Aplica o preenchimento de moda para colunas categóricas.
    for coluna in colunas_categoricas_moda:
        if coluna in dados.columns:
            dados = preencher_moda(dados, coluna)
        else:
            print(f"Aviso: Coluna categórica '{coluna}' não encontrada. Ignorando preenchimento por moda.")

    # Aplica o preenchimento de mediana para colunas numéricas.
    for coluna in colunas_numericas_mediana:
        if coluna in dados.columns:
            dados = preencher_mediana(dados, coluna)
        else:
            print(f"Aviso: Coluna numérica '{coluna}' não encontrada. Ignorando preenchimento por mediana.")

    # Aplica o preenchimento de moda para colunas de score.
    for coluna in colunas_score_moda:
        if coluna in dados.columns:
            dados = preencher_moda(dados, coluna)
        else:
            print(f"Aviso: Coluna de score '{coluna}' não encontrada. Ignorando preenchimento por moda.")

    print("> Pré-processamento concluído.")
    return dados