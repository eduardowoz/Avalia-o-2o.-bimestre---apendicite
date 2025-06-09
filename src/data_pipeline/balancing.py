# src/data_pipeline/balancing.py

import pandas as pd
from imblearn.over_sampling import SMOTE

def balancear(dados: pd.DataFrame, coluna_target: str) -> pd.DataFrame:
    """
    Realiza oversampling em uma coluna alvo usando SMOTE.
    """
    print(f"> Balanceando dados para a coluna '{coluna_target}'...")

    # Define classes a serem removidas para o balanceamento de 'Management'.
    if coluna_target == 'Management':
        classes_para_remover = ['secondary surgical', 'simultaneous appendectomy']
        
        if 'Management' in dados.columns:
            indices_para_remover = dados[dados['Management'].isin(classes_para_remover)].index
            dados_filtrados = dados.drop(indices_para_remover)
            print(f"  Removidas {len(indices_para_remover)} linhas para o balanceamento de 'Management' (classes: {classes_para_remover}).")
        else:
            dados_filtrados = dados.copy()
            print(f"  Aviso: Coluna 'Management' não encontrada, não foi possível filtrar classes.")
    else:
        dados_filtrados = dados.copy()

    # Separa features e a coluna alvo.
    colunas_target_outras = ['Diagnosis', 'Severity', 'Management']
    colunas_a_ignorar_features = [col for col in colunas_target_outras if col != coluna_target]
    
    dados_atributos = dados_filtrados.drop(columns=colunas_a_ignorar_features + [coluna_target], errors='ignore')
    dados_classes = dados_filtrados[coluna_target]

    # Aplica o SMOTE para balanceamento.
    resampler = SMOTE(random_state=42)
    dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)

    # Recompõe o DataFrame balanceado.
    dados_atributos_b.reset_index(drop=True, inplace=True)
    dados_classes_b.reset_index(drop=True, inplace=True)
    dados_final = pd.concat([dados_atributos_b, dados_classes_b], axis=1)

    print(f"> Balanceamento para '{coluna_target}' concluído. Nova distribuição de classes:")
    print(dados_final[coluna_target].value_counts())
    
    return dados_final