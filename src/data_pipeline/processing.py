# src/data_pipeline/processing.py

import pandas as pd
from ucimlrepo import fetch_ucirepo

def obter_conjunto_de_dados_brutos():
    """
    Adquire o conjunto de dados primário de apendicite pediátrica.
    Acessa o repositório UCI ML para obter os dados de features e targets.
    """
    print("> Carregando o conjunto de dados principal...")
    try:
        repositorio_apendicite = fetch_ucirepo(id=938)
        dados_features = repositorio_apendicite.data.features
        dados_alvo = repositorio_apendicite.data.targets
        
        conjunto_completo_dados = pd.concat([dados_features, dados_alvo], axis=1)
        
        print("> Conjunto de dados carregado com sucesso.")

    except Exception as e:
        print(f"ERRO ao carregar o conjunto de dados: {e}")
        print("Verifique a conexão de rede ou a disponibilidade do repositório.")
        return None

    return conjunto_completo_dados

def preencher_por_ocorrencia_comum(estrutura_dados, nome_coluna):
    """
    Preenche valores ausentes em uma coluna usando o valor mais frequente (moda).
    Aplica-se a colunas categóricas ou de scores com valores discretos.
    """
    if nome_coluna in estrutura_dados.columns and not estrutura_dados[nome_coluna].isnull().all():
        valor_comum = estrutura_dados[nome_coluna].mode()[0]
        estrutura_dados[nome_coluna] = estrutura_dados[nome_coluna].fillna(valor_comum)
    elif nome_coluna not in estrutura_dados.columns:
        print(f"Aviso: Coluna '{nome_coluna}' não encontrada para preenchimento por ocorrência comum.")
    return estrutura_dados

def preencher_por_valor_central(estrutura_dados, nome_coluna):
    """
    Preenche valores ausentes em uma coluna numérica usando a mediana.
    Adequado para dados contínuos ou ordinais.
    """
    if nome_coluna in estrutura_dados.columns and pd.api.types.is_numeric_dtype(estrutura_dados[nome_coluna]) and not estrutura_dados[nome_coluna].isnull().all():
        valor_central = estrutura_dados[nome_coluna].median()
        estrutura_dados[nome_coluna] = estrutura_dados[nome_coluna].fillna(valor_central)
    elif nome_coluna not in estrutura_dados.columns:
        print(f"Aviso: Coluna '{nome_coluna}' não encontrada para preenchimento por valor central.")
    elif not pd.api.types.is_numeric_dtype(estrutura_dados[nome_coluna]):
        print(f"Aviso: Coluna '{nome_coluna}' não é numérica. Preenchimento por mediana ignorado.")
    return estrutura_dados

def preparar_dados_para_analise(estrutura_dados_brutos):
    """
    Executa a etapa inicial de preparação dos dados, incluindo:
    - Exclusão de colunas irrelevantes ou incompletas.
    - Tratamento de valores ausentes por imputação.
    """
    print("> Iniciando a preparação dos dados...")

    identificadores_para_descarte = [
        'Segmented_Neutrophils', 'Appendix_Wall_Layers', 'Target_Sign', 'Appendicolith',
        'Perfusion', 'Perforation', 'Surrounding_Tissue_Reaction', 'Appendicular_Abscess',
        'Abscess_Location', 'Pathological_Lymph_Nodes', 'Lymph_Nodes_Location',
        'Bowel_Wall_Thickening', 'Conglomerate_of_Bowel_Loops', 'Ileus', 'Coprostasis',
        'Meteorism', 'Enteritis', 'Gynecological_Findings'
    ]

    colunas_a_descartar_existentes = [col for col in identificadores_para_descarte if col in estrutura_dados_brutos.columns]
    estrutura_dados_processados = estrutura_dados_brutos.drop(columns=colunas_a_descartar_existentes, errors='ignore')
    print(f"Colunas descartadas: {colunas_a_descartar_existentes}")

    categorias_a_preencher_por_moda = [
        'Sex', 'Neutrophilia', 'Ketones_in_Urine', 'Stool',
        'Contralateral_Rebound_Tenderness', 'Coughing_Pain', 'Nausea', 'Loss_of_Appetite',
        'RBC_in_Urine', 'WBC_in_Urine', 'Dysuria', 'Peritonitis', 'Psoas_Sign',
        'Ipsilateral_Rebound_Tenderness', 'US_Performed', 'Free_Fluids',
        'Appendix_on_US', 'Migratory_Pain', 'Lower_Right_Abd_Pain',
        'Diagnosis', 'Severity', 'Management'
    ]

    numericos_a_preencher_por_mediana = [
        'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay', 'Appendix_Diameter',
        'Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage', 'RBC_Count',
        'Hemoglobin', 'RDW', 'Thrombocyte_Count', 'CRP'
    ]

    pontuacoes_a_preencher_por_moda = ['Alvarado_Score', 'Paedriatic_Appendicitis_Score']

    for coluna in categorias_a_preencher_por_moda:
        if coluna in estrutura_dados_processados.columns:
            estrutura_dados_processados = preencher_por_ocorrencia_comum(estrutura_dados_processados, coluna)
        else:
            print(f"Aviso: Categoria '{coluna}' não encontrada. Preenchimento ignorado.")

    for coluna in numericos_a_preencher_por_mediana:
        if coluna in estrutura_dados_processados.columns:
            estrutura_dados_processados = preencher_por_valor_central(estrutura_dados_processados, coluna)
        else:
            print(f"Aviso: Métrica numérica '{coluna}' não encontrada. Preenchimento ignorado.")

    for coluna in pontuacoes_a_preencher_por_moda:
        if coluna in estrutura_dados_processados.columns:
            estrutura_dados_processados = preencher_por_ocorrencia_comum(estrutura_dados_processados, coluna)
        else:
            print(f"Aviso: Pontuação '{coluna}' não encontrada. Preenchimento ignorado.")

    print("> Preparação de dados concluída.")
    return estrutura_dados_processados