# Sistema de Diagnóstico de Apendicite Pediátrica

Este projeto implementa um sistema inteligente para auxiliar no diagnóstico, determinação de severidade e manejo de apendicite pediátrica, utilizando modelos de Machine Learning (RandomForests).

## Estrutura do Projeto

Avalia-o-2o.-bimestre---apendicite/
├── data/
│   └── app_data.csv        # Dataset principal
│   └── pacientes_inferidos.csv # Saída das inferências
├── models/                 # Modelos de ML treinados e scaler
│   └── (arquivos .pkl gerados após o treinamento)
├── src/
│   ├── app/
│   │   ├── init.py
│   │   ├── inference.py    # Lógica de inferência
│   │   ├── menu.py         # Funções de menu e interação
│   │   └── patient_intake.py # Coleta de dados do paciente
│   │
│   ├── data_pipeline/
│   │   ├── init.py
│   │   ├── balancing.py    # Balanceamento de classes (SMOTE)
│   │   ├── normalization.py # Normalização e One-Hot Encoding
│   │   └── processing.py   # Pré-processamento e tratamento de nulos
│   │
│   └── training_pipeline/
│       ├── init.py
│       └── train.py        # Pipeline de treinamento dos modelos
│
├── .gitignore              # Ignora arquivos não versionados
└── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo

## Como Usar

Siga os passos abaixo para configurar e executar o sistema.

### 1. Pré-requisitos

* **Python 3.x** instalado.

### 2. Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd seu_projeto
    ```
2.  **Crie e ative um ambiente virtual:**
    * **Windows (PowerShell):**
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```
    * **macOS / Linux (Bash/Zsh):**
        ```bash
        python -m venv .venv
        source .venv/bin/activate
        ```
    * **Solução de Problemas na Ativação (Windows PowerShell):**
        Se encontrar um erro de "execução de scripts desabilitada", abra o PowerShell como **Administrador** e execute:
        ```powershell
        Set-ExecutionPolicy RemoteSigned
        ```
        Confirme com `S` (Sim) e tente ativar o ambiente virtual novamente no seu terminal de usuário.

3.  **Instale as dependências:**
    Com o ambiente virtual ativado, execute:
    ```bash
    pip install -r requirements.txt
    ```
    * Se, após a instalação, persistir um erro de módulo não encontrado (ex: `ModuleNotFoundError`), tente o comando completo que especifica o interpretador Python do ambiente virtual:
        ```bash
        # Substitua [CAMINHO_COMPLETO_DO_SEU_ENV] pelo caminho real do seu ambiente virtual
        # Ex: C:\Users\SeuUsuario\seu_projeto\.venv\Scripts\python.exe
        [CAMINHO_COMPLETO_DO_SEU_ENV]\python.exe -m pip install -r requirements.txt
        ```

### 3. Executar o Sistema

Com o ambiente virtual ativado e as dependências instaladas, execute o script principal a partir da raiz do projeto:

```bash
python src/main.py

O sistema exibirá um menu no terminal com as seguintes opções:

Treinar modelos: Realiza o pré-processamento dos dados, treina os modelos de classificação e os salva na pasta models/. Esta etapa precisa ser executada pelo menos uma vez antes de diagnosticar pacientes.
Diagnosticar um novo paciente: Coleta informações do paciente via terminal, aplica os modelos treinados e exibe um diagnóstico, severidade e manejo. Os resultados são salvos em data/pacientes_inferidos.csv.
Sair: Encerra a aplicação.
