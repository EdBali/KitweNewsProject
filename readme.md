# News Aggregation Pipeline for Fake News Detection

## Overview
This repository contains a modular pipeline for cleaning, categorizing, and detecting fake news in a dataset of news articles. It is built in Python, incorporating text preprocessing, category prediction using keyword-based and machine-learning-based models, and a robust approach to assess credibility and sensationalism in news content.

The pipeline aims to process news data with minimal human oversight, providing a well-defined and interpretable outcome regarding the credibility of news sources.

## Current Pipeline Modules
### 1. **Data Cleaning** (`data_cleaning.py`)
   - Handles loading, cleaning, and standardizing text-based data.
   - Processes and normalizes text, converts date formats, and removes duplicates.
   - Cleans the text by removing punctuation, lemmatizing words, and filtering stop words.
   
### 2. **Categorization** (`text_categorizer.py`)
   - Predicts categories for each news article by keyword analysis.
   - Uses K-Nearest Neighbors (KNN) for additional category prediction, trained on a TF-IDF representation of descriptions.
   - Outputs a cleaned dataset with well-defined categories.

### 3. **Fake News Detection** (`fake_news_detector.py`)
   - Flags fake news using a combination of heuristic indicators such as source credibility, sensationalism, headline-description mismatch, and excessive capitalization.
   - Combines the indicators into a final "fake news score" to classify articles as credible or fake.

### 4. **Main Program Execution** (`main.py`)
   - Orchestrates the process, loading data, cleaning it, categorizing articles, running fake news detection, and saving the final results.
   
   ```shell
   python main.py
   
## Usage
1. **Clone the Repository:**
   ```shell
   git clone <repository-url>
   cd <repository-directory>

2. **Install Dependencies:**
    ```shell
    pip install -r requirements.txt

3. **Run the pipeline**
    ```shell
    python main.py

## Future Developments
Containerization and Workflow Orchestration
Currently, work is ongoing to containerize each module for more consistent deployment. Orchestration via Airflow is also in progress, which will enable streamlined scheduling and data flow management for the pipeline.

Automated Data Collection (RSS Scraping)
An automated RSS feed scraping tool is being developed as the first stage of the pipeline. Though still in progress, this component will provide an up-to-date stream of news articles, simplifying the pipeline’s integration with various news sources.