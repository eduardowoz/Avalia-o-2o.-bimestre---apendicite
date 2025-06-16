# src/app/menu.py

import os

# Importa as funções para a lógica da aplicação de inferência e coleta de dados
from app.inference import realizar_analise_paciente
from app.patient_intake import registrar_novo_paciente

# Importa a função principal para o processo de treinamento dos modelos
from training_pipeline.train import iniciar_pipeline_de_treinamento

def purificar_terminal():
    """Limpa o console do terminal para melhor visualização."""
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def exibir_opcoes_principais():
    """
    Apresenta o menu de opções para o usuário e coleta sua escolha.
    Retorna a opção numérica selecionada.
    """
    purificar_terminal()
    print("=" * 50)
    print("   Sistema de Apoio ao Diagnóstico de Apendicite")
    print("=" * 50)
    print("\nEscolha uma operação:")
    print("[ 1 ] - Gerar e treinar modelos de predição")
    print("[ 2 ] - Inserir dados para análise de um novo paciente")
    print("[ 0 ] - Encerrar aplicação")
    print("=" * 50)
    
    while True:
        selecao_usuario = input("Sua escolha: ").strip()
        if selecao_usuario.isdigit():
            return int(selecao_usuario)
        else:
            print("Opção inválida. Por favor, digite o número correspondente à sua escolha.")

def executar_fluxo_principal():
    """
    Controla o fluxo de navegação do menu da aplicação.
    Direciona para as funcionalidades de treinamento ou análise de paciente.
    """
    while True:
        opcao_selecionada = exibir_opcoes_principais()

        if opcao_selecionada == 1:
            purificar_terminal()
            iniciar_pipeline_de_treinamento()
            input("\nPressione ENTER para retornar ao menu principal...")
        elif opcao_selecionada == 2:
            dados_paciente_processados = registrar_novo_paciente()
            if dados_paciente_processados is not None:
                purificar_terminal()
                print("=" * 50)
                print("      Relatório de Análise do Paciente")
                print("=" * 50)
                realizar_analise_paciente(dados_paciente_processados)
                print("\n" + "=" * 50)
                input("\nPressione ENTER para retornar ao menu principal...")
            else:
                input("\nOperação cancelada. Pressione ENTER para retornar ao menu...")
        elif opcao_selecionada == 0:
            print("\nEncerrando a aplicação. Obrigado(a)!")
            break
        else:
            purificar_terminal()
            print("Opção não reconhecida. Por favor, tente novamente.")
            input("\nPressione ENTER para continuar...")