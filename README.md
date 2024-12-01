# skill_recommendation


Folder structure:

your_project/
│
├── config/
│   └── config.json            # Configuration file for models, hyperparameters, etc.
│
├── data/
│   ├── market_trends.json     # Market trends data
│   ├── pay_info.csv           # Pay info for different tech careers
│   └── output/                # Folder to store RAG output
│
├── rag_pipeline/              # RAG pipeline code
│   ├── __init__.py
│   ├── rag_pipeline.py        # Main script for the RAG pipeline
│   ├── utils.py               # Helper functions for the RAG pipeline
│   └── requirements.txt       # Dependencies for the RAG pipeline
│
├── recommender_pipeline/      # Recommender pipeline code
│   ├── __init__.py
│   ├── recommender.py         # Main script for the recommender pipeline
│   ├── utils.py               # Helper functions for the recommender pipeline
│   └── requirements.txt       # Dependencies for the recommender pipeline
│
├── common_utils/              # Common utility functions for both pipelines
│   ├── __init__.py
│   ├── data_loader.py         # Functions to load market trends, pay info, etc.
│   ├── model_loader.py        # Functions to load models dynamically
│   └── file_manager.py        # File I/O operations for saving and loading data
│
├── main.py                    # Main script to invoke the pipeline (entry point)
├── requirements.txt           # Global requirements file (for the entire project)
└── README.md                  # Project documentation
