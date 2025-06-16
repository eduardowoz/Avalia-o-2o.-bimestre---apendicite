# src/data_pipeline/balancing.py

import pandas as pd
from imblearn.over_sampling import SMOTE

def ajustar_desbalanceamento_classes(conjunto_de_dados: pd.DataFrame, identificador_alvo: str) -> pd.DataFrame:
    """
    Realiza oversampling (aumento de amostras da classe minoritária) em uma coluna alvo.
    Utiliza a técnica SMOTE para sintetizar novas amostras.
    """
    print(f"> Balanceando dados para o identificador alvo: '{identificador_alvo}'...")

    if identificador_alvo == 'Management':
        opcoes_de_manejo_a_excluir = ['secondary surgical', 'simultaneous appendectomy']
        
        if 'Management' in conjunto_de_dados.columns:
            indices_para_descarte = conjunto_de_dados[conjunto_de_dados['Management'].isin(opcoes_de_manejo_a_excluir)].index
            estrutura_dados_filtrada = conjunto_de_dados.drop(indices_para_descarte)
            print(f"  Descartadas {len(indices_para_descarte)} linhas para balanceamento de 'Management' (opções: {opcoes_de_manejo_a_excluir}).")
        else:
            estrutura_dados_filtrada = conjunto_de_dados.copy()
            print(f"  Aviso: Identificador alvo 'Management' não encontrado, filtragem de opções ignorada.")
    else:
        estrutura_dados_filtrada = conjunto_de_dados.copy()

    outros_identificadores_alvo = ['Diagnosis', 'Severity', 'Management']
    colunas_a_ignorar_nas_features = [col for col in outros_identificadores_alvo if col != identificador_alvo]
    
    atributos_para_balancear = estrutura_dados_filtrada.drop(columns=colunas_a_ignorar_nas_features + [identificador_alvo], errors='ignore')
    classes_para_balancear = estrutura_dados_filtrada[identificador_alvo]

    instancia_reamostrador = SMOTE(random_state=42)
    atributos_reamostrados, classes_reamostradas = instancia_reamostrador.fit_resample(atributos_para_balancear, classes_para_balancear)

    atributos_reamostrados.reset_index(drop=True, inplace=True)
    classes_reamostradas.reset_index(drop=True, inplace=True)
    dados_reamostrados_final = pd.concat([atributos_reamostrados, classes_reamostradas], axis=1)

    print(f"> Balanceamento para '{identificador_alvo}' concluído. Nova distribuição de classes:")
    print(dados_reamostrados_final[identificador_alvo].value_counts())
    
    return dados_reamostrados_final