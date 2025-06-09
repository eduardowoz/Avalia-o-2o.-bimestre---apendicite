# src/app/menu.py

import os # Para limpar a tela

# Importa as funções dos outros módulos da aplicação
from app.inference import inferir_paciente
from app.patient_intake import novo_paciente

# Importa a função principal do pipeline de treinamento
from training_pipeline.train import executar_pipeline_de_treinamento

def limpar_tela():
    """Limpa a tela do terminal."""
    # Para Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # Para Mac e Linux
    else:
        _ = os.system('clear')

def exibir_menu_principal():
    """Exibe o menu principal e retorna a escolha do usuário."""
    limpar_tela()
    print("=" * 50)
    print("   Modelo de Predição de Apendicite Pediátrica")
    print("=" * 50)
    print("\nSelecione uma opção:")
    print("[ 1 ] - Treinar modelos")
    print("[ 2 ] - Diagnosticar um novo paciente")
    print("[ 0 ] - Sair")
    print("=" * 50)
    
    while True:
        escolha = input("Sua escolha: ").strip()
        if escolha.isdigit():
            return int(escolha)
        else:
            print("Opção inválida. Por favor, digite um número correspondente à opção.")

def menu_principal():
    """Gerencia o fluxo do menu principal da aplicação."""
    while True:
        escolha = exibir_menu_principal()

        if escolha == 1:
            limpar_tela()
            executar_pipeline_de_treinamento()
            input("\nPressione ENTER para voltar ao menu principal...")
        elif escolha == 2:
            paciente = novo_paciente() # Coleta os dados do paciente
            if paciente is not None: # Se a coleta não foi cancelada
                limpar_tela()
                print("=" * 50)
                print("      Resultado da Análise do Paciente")
                print("=" * 50)
                inferir_paciente(paciente) # Roda a inferência
                print("\n" + "=" * 50)
                input("\nPressione ENTER para voltar ao menu principal...")
            else: # Se a coleta foi cancelada
                input("\nColeta de dados cancelada. Pressione ENTER para voltar...")
        elif escolha == 0:
            print("\nEncerrando o programa...")
            break
        else:
            limpar_tela()
            print("Opção inválida. Tente novamente.")
            input("\nPressione ENTER para continuar...")