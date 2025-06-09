# src/app/patient_intake.py

import pandas as pd
import sys
import os

# Importa a função de normalização do pipeline de dados
from data_pipeline.normalization import normalizar_paciente

def limpar_tela():
    """Limpa a tela do terminal."""
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def is_float(text):
    """Valida se uma string pode ser convertida para float."""
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False

def coletar_dado(pergunta, tipo='str', opcoes=None, valor_default=None):
    """
    Coleta uma entrada do usuário com validação.
    Retorna None se a coleta for cancelada.
    """
    while True:
        try:
            if opcoes:
                print(f"\n{pergunta} Opções: {', '.join(opcoes)}")
                resposta = input("Sua escolha: ").strip()
                if resposta == '' and valor_default is not None:
                    return valor_default
                if resposta not in opcoes:
                    print("Opção inválida. Por favor, escolha uma das opções fornecidas.")
                    continue
            else:
                resposta = input(f"\n{pergunta}: ").strip()
                if resposta == '' and valor_default is not None:
                    return valor_default
            
            if resposta == '' and tipo != 'str' and valor_default is None:
                print("Entrada obrigatória não pode ser vazia.")
                continue

            if tipo == 'float':
                if is_float(resposta):
                    return float(resposta)
                else:
                    print("Entrada inválida. Por favor, digite um número (ex: 37.5 ou 10).")
                    continue
            elif tipo == 'int':
                if resposta.isdigit():
                    return int(resposta)
                else:
                    print("Entrada inválida. Por favor, digite um número inteiro.")
                    continue
            elif tipo == 'bool_sim_nao':
                if resposta.lower() == 'sim': return 'yes'
                if resposta.lower() == 'nao' or resposta.lower() == 'não': return 'no'
                print("Entrada inválida. Por favor, digite 'Sim' ou 'Não'.")
                continue
            elif tipo == 'genero':
                if resposta.lower() == 'feminino': return 'female'
                if resposta.lower() == 'masculino': return 'male'
                print("Entrada inválida. Por favor, digite 'Feminino' ou 'Masculino'.")
                continue
            elif tipo == 'fezes':
                if resposta.lower() == 'normal': return 'normal'
                if resposta.lower() == 'constipacao' or resposta.lower() == 'constipação': return 'constipation'
                if resposta.lower() == 'diarreia': return 'diarrhea'
                if resposta.lower() == 'constipacao e diarreia' or resposta.lower() == 'constipação e diarreia': return 'constipation, diarrhea'
                print("Entrada inválida. Opções: Normal, Constipação, Diarreia, Constipação e Diarreia.")
                continue
            elif tipo == 'peritonite':
                if resposta.lower() == 'nao' or resposta.lower() == 'não': return 'no'
                if resposta.lower() == 'localizada': return 'local'
                if resposta.lower() == 'generalizada': return 'generalized'
                print("Entrada inválida. Opções: Não, Localizada, Generalizada.")
                continue
            elif tipo == 'cetonas' or tipo == 'rbc_wbc_urina':
                if resposta.lower() == 'no' or resposta.lower() == 'não': return 'no'
                if resposta == '+': return '+'
                if resposta == '++': return '++'
                if resposta == '+++': return '+++'
                print("Entrada inválida. Opções: Não, +, ++, +++.")
                continue
            else:
                return resposta.lower() if isinstance(resposta, str) else resposta
        except KeyboardInterrupt:
            print("\nColeta de dados cancelada pelo usuário.")
            return None

def coletar_dados_paciente():
    """Coleta interativamente os dados de um novo paciente."""
    limpar_tela()
    print("=" * 50)
    print("     Coleta de Dados do Novo Paciente")
    print("=" * 50)
    print("\nInstruções:")
    print(" -> Preencha os campos solicitados.")
    print(" -> Em campos com 'Sim'/'Não', digite a palavra completa.")
    print(" -> Pressione Ctrl+C a qualquer momento para cancelar a coleta.\n")

    dados_paciente = {}

    # Configurações para cada pergunta.
    perguntas_config = {
        'Age': {'pergunta': 'Qual a idade do paciente?', 'tipo': 'float'},
        'BMI': {'pergunta': 'Qual o IMC (Índice de Massa Corporal)?', 'tipo': 'float'},
        'Sex': {'pergunta': 'Qual o sexo do paciente?', 'tipo': 'genero', 'opcoes': ['Feminino', 'Masculino']},
        'Height': {'pergunta': 'Qual a altura (em cm)?', 'tipo': 'float'},
        'Weight': {'pergunta': 'Qual o peso (em kg)?', 'tipo': 'float'},
        'Length_of_Stay': {'pergunta': 'Qual foi o tempo de permanência (em dias)?', 'tipo': 'float'},
        
        'Migratory_Pain': {'pergunta': 'Houve dor migratória?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Lower_Right_Abd_Pain': {'pergunta': 'Paciente tem dor no abdômen inferior direito?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Contralateral_Rebound_Tenderness': {'pergunta': 'Houve sensibilidade contralateral (Blumberg)?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Coughing_Pain': {'pergunta': 'Sente dor ao tossir?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Nausea': {'pergunta': 'Apresentou náusea?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Loss_of_Appetite': {'pergunta': 'Houve perda de apetite?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Body_Temperature': {'pergunta': 'Qual a temperatura corporal (em °C)?', 'tipo': 'float'},
        'Dysuria': {'pergunta': 'Apresenta disúria (dor ao urinar)?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Stool': {'pergunta': 'Como está o trânsito intestinal (fezes)?', 'tipo': 'fezes', 'opcoes': ['Normal', 'Constipação', 'Diarreia', 'Constipação e Diarreia']},
        'Peritonitis': {'pergunta': 'Apresenta sinais de peritonite?', 'tipo': 'peritonite', 'opcoes': ['Não', 'Localizada', 'Generalizada']},
        'Psoas_Sign': {'pergunta': 'O sinal de Psoas é positivo?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Ipsilateral_Rebound_Tenderness': {'pergunta': 'Houve sensibilidade ipsilateral?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        
        'Alvarado_Score': {'pergunta': 'Qual o Escore de Alvarado (0-10)?', 'tipo': 'int', 'opcoes': [str(i) for i in range(11)]},
        'Paedriatic_Appendicitis_Score': {'pergunta': 'Qual o Escore Pediátrico de Apendicite (PAS) (0-10)?', 'tipo': 'int', 'opcoes': [str(i) for i in range(11)]},
        
        'WBC_Count': {'pergunta': 'Qual a contagem de leucócitos (x10^9/L)?', 'tipo': 'float'},
        'Neutrophil_Percentage': {'pergunta': 'Qual a porcentagem de neutrófilos (%)?', 'tipo': 'float'},
        'Neutrophilia': {'pergunta': 'Apresenta neutrofilia?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'RBC_Count': {'pergunta': 'Qual a contagem de hemácias?', 'tipo': 'float'},
        'Hemoglobin': {'pergunta': 'Qual o nível de hemoglobina (g/dL)?', 'tipo': 'float'},
        'RDW': {'pergunta': 'Qual o valor do RDW (%)?', 'tipo': 'float'},
        'Thrombocyte_Count': {'pergunta': 'Qual a contagem de plaquetas?', 'tipo': 'float'},
        'Ketones_in_Urine': {'pergunta': 'Qual o nível de cetonas na urina?', 'tipo': 'cetonas', 'opcoes': ['Não', '+', '++', '+++']},
        'RBC_in_Urine': {'pergunta': 'Qual o nível de hemácias na urina?', 'tipo': 'rbc_wbc_urina', 'opcoes': ['Não', '+', '++', '+++']},
        'WBC_in_Urine': {'pergunta': 'Qual o nível de leucócitos na urina?', 'tipo': 'rbc_wbc_urina', 'opcoes': ['Não', '+', '++', '+++']},
        'CRP': {'pergunta': 'Qual o valor da Proteína C-Reativa (PCR) (mg/L)?', 'tipo': 'float'},
        
        'US_Performed': {'pergunta': 'O ultrassom foi realizado?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Appendix_on_US': {'pergunta': 'O apêndice foi visível no ultrassom?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
        'Appendix_Diameter': {'pergunta': 'Qual o diâmetro do apêndice (em mm)?', 'tipo': 'float'},
        'Free_Fluids': {'pergunta': 'Foram observados fluidos livres no ultrassom?', 'tipo': 'bool_sim_nao', 'opcoes': ['Sim', 'Não']},
    }

    # Ordem e agrupamento das perguntas.
    ordem_perguntas = [
        'Age', 'Sex', 'Height', 'Weight', 'BMI', 'Length_of_Stay',
        'Migratory_Pain', 'Lower_Right_Abd_Pain', 'Contralateral_Rebound_Tenderness',
        'Coughing_Pain', 'Nausea', 'Loss_of_Appetite', 'Body_Temperature', 'Dysuria',
        'Stool', 'Peritonitis', 'Psoas_Sign', 'Ipsilateral_Rebound_Tenderness',
        'Alvarado_Score', 'Paedriatic_Appendicitis_Score',
        'WBC_Count', 'Neutrophil_Percentage', 'Neutrophilia', 'RBC_Count', 'Hemoglobin',
        'RDW', 'Thrombocyte_Count', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine', 'CRP',
        'US_Performed'
    ]

    # Coleta de dados.
    for col in ordem_perguntas:
        config = perguntas_config[col]
        pergunta = config['pergunta']
        tipo = config['tipo']
        opcoes = config.get('opcoes')

        if col == 'US_Performed':
            us_performed = coletar_dado(pergunta, tipo, opcoes)
            if us_performed is None: return None
            dados_paciente[col] = us_performed

            # Se ultrassom não foi realizado, preenche dados relacionados com valores padrão.
            if us_performed == 'no':
                dados_paciente['Appendix_on_US'] = 'no'
                dados_paciente['Appendix_Diameter'] = 0.0
                dados_paciente['Free_Fluids'] = 'no'
            else: # Se ultrassom foi realizado, coleta mais informações.
                res_app_on_us = coletar_dado(perguntas_config['Appendix_on_US']['pergunta'], 
                                             perguntas_config['Appendix_on_US']['tipo'], 
                                             perguntas_config['Appendix_on_US']['opcoes'])
                if res_app_on_us is None: return None
                dados_paciente['Appendix_on_US'] = res_app_on_us

                res_app_diameter = coletar_dado(perguntas_config['Appendix_Diameter']['pergunta'], 
                                                perguntas_config['Appendix_Diameter']['tipo'])
                if res_app_diameter is None: return None
                dados_paciente['Appendix_Diameter'] = res_app_diameter

                res_free_fluids = coletar_dado(perguntas_config['Free_Fluids']['pergunta'], 
                                              perguntas_config['Free_Fluids']['tipo'], 
                                              perguntas_config['Free_Fluids']['opcoes'])
                if res_free_fluids is None: return None
                dados_paciente['Free_Fluids'] = res_free_fluids
            
            continue # Pula para a próxima pergunta após tratar o ultrassom.

        # Coleta dados para as outras perguntas.
        resposta = coletar_dado(pergunta, tipo, opcoes)
        if resposta is None:
            return None
        dados_paciente[col] = resposta
        
    return dados_paciente

def novo_paciente():
    """Coleta dados do paciente e os normaliza para inferência."""
    dados_dict = coletar_dados_paciente()
    
    if dados_dict is None:
        print("\nColeta de dados cancelada. Retornando ao menu principal.")
        return None

    # Converte os dados coletados em DataFrame e os normaliza.
    paciente_df = pd.DataFrame([dados_dict])
    paciente_normalizado = normalizar_paciente(paciente_df)
    
    return paciente_normalizado