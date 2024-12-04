# Skill Recommendation System

This project is a skill recommendation system designed to assist gig workers and freelancers in the tech industry. It uses advanced AI models, such as language models (LLMs) and Retrieval-Augmented Generation (RAG), to provide personalized career guidance based on users' current skills, market trends, and income data. 

The system identifies in-demand skills, maps transferable skills to relevant job roles, compares income potential, and offers upskilling recommendations to help users achieve their career goals.

---

## Features

- **Skill Mapping**: Identifies in-demand skills the user already possesses and maps them to transferable skills for high-demand job roles.
- **Income Comparison**: Compares the user’s current income with market trends to highlight roles with better earning potential.
- **Career Recommendations**: Suggests the best career paths for the user based on their skills and income data.
- **Upskilling Directions**: Provides a step-by-step guide for users to acquire the necessary skills for a career switch.

---

## Folder Structure

The project is modular and organized into the following folders:

### **`config/`**
- **`config.json`**: Contains configurations for the models, hyperparameters, and pipeline settings.

### **`data/`**
- **`market_trends.json`**: Stores data on in-demand skills, job roles, and market trends in the tech industry.
- **`pay_info.csv`**: Contains pay information for various tech careers.
- **`output/`**: Directory for storing outputs generated by the RAG pipeline.

### **`rag_pipeline/`**
- **`__init__.py`**: Marks the folder as a Python module.
- **`rag_pipeline.py`**: The main script implementing the RAG pipeline, retrieving relevant market insights for the recommendation system.
- **`utils.py`**: Includes helper functions, such as data pre-processing and response formatting.
- **`requirements.txt`**: Dependencies specific to the RAG pipeline.

### **`recommender_pipeline/`**
- **`__init__.py`**: Marks the folder as a Python module.
- **`recommender.py`**: Implements the recommendation pipeline, including skill mapping, income comparison, and career recommendations.
- **`utils.py`**: Provides utility functions for processing recommendations and insights.
- **`requirements.txt`**: Dependencies for the recommender pipeline.

### **`common_utils/`**
- **`__init__.py`**: Marks the folder as a Python module.
- **`data_loader.py`**: Handles loading of data from files (e.g., `market_trends.json`, `pay_info.csv`).
- **`model_loader.py`**: Dynamically loads LLMs and other models used in the pipelines.
- **`file_manager.py`**: Manages file operations, including saving and retrieving output data.

### **Project Root Files**
- **`main.py`**: The entry point for running the project. It integrates the RAG and recommendation pipelines.
- **`requirements.txt`**: A global requirements file for installing dependencies across the entire project.
- **`README.md`**: This documentation file, explaining the project setup and functionality.

---

## How to Use

1. **Install Dependencies**:
   Install all required Python packages:
   ```bash
   pip install -r requirements.txt
