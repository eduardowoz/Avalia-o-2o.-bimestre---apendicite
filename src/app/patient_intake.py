# src/app/patient_intake.py

import pandas as pd
import sys
import os

# Importa a função de preparação de dados para inferência
from data_pipeline.normalization import preparar_paciente_para_inferencia

def apagar_tela_terminal():
    """Limpa o conteúdo visível do terminal."""
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def eh_numero_decimal(texto):
    """Verifica se uma string pode ser convertida para um número decimal."""
    if not texto:
        return False
    try:
        float(texto)
        return True
    except ValueError:
        return False

def obter_entrada_usuario(questionamento, tipo_esperado='str', opcoes_validas=None, valor_ausente=None):
    """
    Solicita e valida uma entrada do usuário.
    Retorna None se o usuário optar por cancelar a entrada.
    """
    while True:
        try:
            if opcoes_validas:
                print(f"\n{questionamento} Opções: {', '.join(opcoes_validas)}")
                resposta_bruta = input("Sua escolha: ").strip()
                if resposta_bruta == '' and valor_ausente is not None:
                    return valor_ausente
                if resposta_bruta not in opcoes_validas:
                    print("Opção inválida. Por favor, escolha uma das opções fornecidas.")
                    continue
            else:
                resposta_bruta = input(f"\n{questionamento}: ").strip()
                if resposta_bruta == '' and valor_ausente is not None:
                    return valor_ausente
            
            if resposta_bruta == '' and tipo_esperado != 'str' and valor_ausente is None:
                print("Entrada obrigatória não pode ser vazia.")
                continue

            if tipo_esperado == 'float':
                if eh_numero_decimal(resposta_bruta):
                    return float(resposta_bruta)
                else:
                    print("Entrada inválida. Por favor, digite um número (ex: 37.5 ou 10).")
                    continue
            elif tipo_esperado == 'int':
                if resposta_bruta.isdigit():
                    return int(resposta_bruta)
                else:
                    print("Entrada inválida. Por favor, digite um número inteiro.")
                    continue
            elif tipo_esperado == 'resposta_sim_nao':
                if resposta_bruta.lower() == 'sim': return 'yes'
                if resposta_bruta.lower() == 'nao' or resposta_bruta.lower() == 'não': return 'no'
                print("Entrada inválida. Por favor, digite 'Sim' ou 'Não'.")
                continue
            elif tipo_esperado == 'classificacao_genero':
                if resposta_bruta.lower() == 'feminino': return 'female'
                if resposta_bruta.lower() == 'masculino': return 'male'
                print("Entrada inválida. Por favor, digite 'Feminino' ou 'Masculino'.")
                continue
            elif tipo_esperado == 'estado_fezes':
                if resposta_bruta.lower() == 'normal': return 'normal'
                if resposta_bruta.lower() == 'constipacao' or resposta_bruta.lower() == 'constipação': return 'constipation'
                if resposta_bruta.lower() == 'diarreia': return 'diarrhea'
                if resposta_bruta.lower() == 'constipacao e diarreia' or resposta_bruta.lower() == 'constipação e diarreia': return 'constipation, diarrhea'
                print("Entrada inválida. Opções: Normal, Constipação, Diarreia, Constipação e Diarreia.")
                continue
            elif tipo_esperado == 'tipo_peritonite':
                if resposta_bruta.lower() == 'nao' or resposta_bruta.lower() == 'não': return 'no'
                if resposta_bruta.lower() == 'localizada': return 'local'
                if resposta_bruta.lower() == 'generalizada': return 'generalized'
                print("Entrada inválida. Opções: Não, Localizada, Generalizada.")
                continue
            elif tipo_esperado == 'niveis_urina':
                if resposta_bruta.lower() == 'no' or resposta_bruta.lower() == 'não': return 'no'
                if resposta_bruta == '+': return '+'
                if resposta_bruta == '++': return '++'
                if resposta_bruta == '+++': return '+++'
                print("Entrada inválida. Opções: Não, +, ++, +++.")
                continue
            else:
                return resposta_bruta.lower() if isinstance(resposta_bruta, str) else resposta_bruta
        except KeyboardInterrupt:
            print("\nColeta de dados cancelada pelo usuário.")
            return None

def coletar_informacoes_do_paciente():
    """
    Guia o usuário na inserção das informações clínicas de um novo paciente.
    Retorna um dicionário com todos os dados coletados ou None em caso de cancelamento.
    """
    apagar_tela_terminal()
    print("=" * 50)
    print("     Registro de Informações do Paciente")
    print("=" * 50)
    print("\nInstruções:")
    print(" -> Preencha os campos solicitados.")
    print(" -> Para perguntas com 'Sim'/'Não', digite a palavra completa.")
    print(" -> Pressione Ctrl+C a qualquer momento para cancelar a operação.\n")

    dados_paciente_compilados = {}

    configuracoes_perguntas = {
        'Age': {'questionamento': 'Qual a idade do paciente?', 'tipo_esperado': 'float'},
        'BMI': {'questionamento': 'Qual o IMC (Índice de Massa Corporal)?', 'tipo_esperado': 'float'},
        'Sex': {'questionamento': 'Qual o sexo do paciente?', 'tipo_esperado': 'classificacao_genero', 'opcoes_validas': ['Feminino', 'Masculino']},
        'Height': {'questionamento': 'Qual a altura (em cm)?', 'tipo_esperado': 'float'},
        'Weight': {'questionamento': 'Qual o peso (em kg)?', 'tipo_esperado': 'float'},
        'Length_of_Stay': {'questionamento': 'Qual foi o tempo de permanência no hospital (em dias)?', 'tipo_esperado': 'float'},
        
        'Migratory_Pain': {'questionamento': 'Houve dor migratória?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Lower_Right_Abd_Pain': {'questionamento': 'O paciente tem dor no abdômen inferior direito?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Contralateral_Rebound_Tenderness': {'questionamento': 'Foi observada sensibilidade contralateral (sinal de Blumberg)?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Coughing_Pain': {'questionamento': 'O paciente sente dor ao tossir?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Nausea': {'questionamento': 'O paciente apresentou náusea?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Loss_of_Appetite': {'questionamento': 'Houve perda de apetite?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Body_Temperature': {'questionamento': 'Qual a temperatura corporal (em °C)?', 'tipo_esperado': 'float'},
        'Dysuria': {'questionamento': 'O paciente apresenta disúria (dor ao urinar)?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Stool': {'questionamento': 'Como está o trânsito intestinal (fezes)?', 'tipo_esperado': 'estado_fezes', 'opcoes_validas': ['Normal', 'Constipação', 'Diarreia', 'Constipação e Diarreia']},
        'Peritonitis': {'questionamento': 'Foram observados sinais de peritonite?', 'tipo_esperado': 'tipo_peritonite', 'opcoes_validas': ['Não', 'Localizada', 'Generalizada']},
        'Psoas_Sign': {'questionamento': 'O sinal de Psoas é positivo?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Ipsilateral_Rebound_Tenderness': {'questionamento': 'Houve sensibilidade ipsilateral?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        
        'Alvarado_Score': {'questionamento': 'Qual o Escore de Alvarado (0-10)?', 'tipo_esperado': 'int', 'opcoes_validas': [str(i) for i in range(11)]},
        'Paedriatic_Appendicitis_Score': {'questionamento': 'Qual o Escore Pediátrico de Apendicite (PAS) (0-10)?', 'tipo_esperado': 'int', 'opcoes_validas': [str(i) for i in range(11)]},
        
        'WBC_Count': {'questionamento': 'Qual a contagem de leucócitos (x10^9/L)?', 'tipo_esperado': 'float'},
        'Neutrophil_Percentage': {'questionamento': 'Qual a porcentagem de neutrófilos (%)?', 'tipo_esperado': 'float'},
        'Neutrophilia': {'questionamento': 'O paciente apresenta neutrofilia?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'RBC_Count': {'questionamento': 'Qual a contagem de hemácias?', 'tipo_esperado': 'float'},
        'Hemoglobin': {'questionamento': 'Qual o nível de hemoglobina (g/dL)?', 'tipo_esperado': 'float'},
        'RDW': {'questionamento': 'Qual o valor do RDW (%)?', 'tipo_esperado': 'float'},
        'Thrombocyte_Count': {'questionamento': 'Qual a contagem de plaquetas?', 'tipo_esperado': 'float'},
        'Ketones_in_Urine': {'questionamento': 'Qual o nível de cetonas na urina?', 'tipo_esperado': 'niveis_urina', 'opcoes_validas': ['Não', '+', '++', '+++']},
        'RBC_in_Urine': {'questionamento': 'Qual o nível de hemácias na urina?', 'tipo_esperado': 'niveis_urina', 'opcoes_validas': ['Não', '+', '++', '+++']},
        'WBC_in_Urine': {'questionamento': 'Qual o nível de leucócitos na urina?', 'tipo_esperado': 'niveis_urina', 'opcoes_validas': ['Não', '+', '++', '+++']},
        'CRP': {'questionamento': 'Qual o valor da Proteína C-Reativa (PCR) (mg/L)?', 'tipo_esperado': 'float'},
        
        'US_Performed': {'questionamento': 'O ultrassom foi realizado?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Appendix_on_US': {'questionamento': 'O apêndice foi visível no ultrassom?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
        'Appendix_Diameter': {'questionamento': 'Qual o diâmetro do apêndice (em mm)?', 'tipo_esperado': 'float'},
        'Free_Fluids': {'questionamento': 'Foram observados fluidos livres no ultrassom?', 'tipo_esperado': 'resposta_sim_nao', 'opcoes_validas': ['Sim', 'Não']},
    }

    sequencia_perguntas = [
        'Age', 'Sex', 'Height', 'Weight', 'BMI', 'Length_of_Stay',
        'Migratory_Pain', 'Lower_Right_Abd_Pain', 'Contralateral_Rebound_Tenderness',
        'Coughing_Pain', 'Nausea', 'Loss_of_Appetite', 'Body_Temperature', 'Dysuria',
        'Stool', 'Peritonitis', 'Psoas_Sign', 'Ipsilateral_Rebound_Tenderness',
        'Alvarado_Score', 'Paedriatic_Appendicitis_Score',
        'WBC_Count', 'Neutrophil_Percentage', 'Neutrophilia', 'RBC_Count', 'Hemoglobin',
        'RDW', 'Thrombocyte_Count', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine', 'CRP',
        'US_Performed'
    ]

    for coluna_dado in sequencia_perguntas:
        config_pergunta = configuracoes_perguntas[coluna_dado]
        questionamento_atual = config_pergunta['questionamento']
        tipo_dado_esperado = config_pergunta['tipo_esperado']
        opcoes_disponiveis = config_pergunta.get('opcoes_validas')

        if coluna_dado == 'US_Performed':
            ultrassom_realizado = obter_entrada_usuario(questionamento_atual, tipo_dado_esperado, opcoes_disponiveis)
            if ultrassom_realizado is None: return None
            dados_paciente_compilados[coluna_dado] = ultrassom_realizado

            if ultrassom_realizado == 'no':
                dados_paciente_compilados['Appendix_on_US'] = 'no'
                dados_paciente_compilados['Appendix_Diameter'] = 0.0
                dados_paciente_compilados['Free_Fluids'] = 'no'
            else:
                info_apendice_us = obter_entrada_usuario(configuracoes_perguntas['Appendix_on_US']['questionamento'], 
                                             configuracoes_perguntas['Appendix_on_US']['tipo_esperado'], 
                                             configuracoes_perguntas['Appendix_on_US']['opcoes_validas'])
                if info_apendice_us is None: return None
                dados_paciente_compilados['Appendix_on_US'] = info_apendice_us

                diametro_apendice = obter_entrada_usuario(configuracoes_perguntas['Appendix_Diameter']['questionamento'], 
                                                 configuracoes_perguntas['Appendix_Diameter']['tipo_esperado'])
                if diametro_apendice is None: return None
                dados_paciente_compilados['Appendix_Diameter'] = diametro_apendice

                fluidos_livres = obter_entrada_usuario(configuracoes_perguntas['Free_Fluids']['questionamento'], 
                                              configuracoes_perguntas['Free_Fluids']['tipo_esperado'], 
                                              configuracoes_perguntas['Free_Fluids']['opcoes_validas'])
                if fluidos_livres is None: return None
                dados_paciente_compilados['Free_Fluids'] = fluidos_livres
            
            continue

        resposta_coletada = obter_entrada_usuario(questionamento_atual, tipo_dado_esperado, opcoes_disponiveis)
        if resposta_coletada is None:
            return None
        dados_paciente_compilados[coluna_dado] = resposta_coletada
        
    return dados_paciente_compilados

def registrar_novo_paciente():
    """
    Orquestra a coleta de informações de um novo paciente e a preparação
    desses dados para serem utilizados pelos modelos de inferência.
    """
    dicionario_dados_brutos = coletar_informacoes_do_paciente()
    
    if dicionario_dados_brutos is None:
        print("\nRegistro do paciente cancelado. Retornando ao menu principal.")
        return None

    dataframe_paciente = pd.DataFrame([dicionario_dados_brutos])
    paciente_pronto_para_modelo = preparar_paciente_para_inferencia(dataframe_paciente)
    
    return paciente_pronto_para_modelo