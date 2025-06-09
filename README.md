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