# Parent Paper Reproduction

## Paper
Impact of Music on Brain and Mental Health: Classification using Machine Learning Algorithms

## Objective
The goal of this module is to reproduce the results of the parent paper using the original dataset and a comparable machine learning pipeline. This serves as the baseline implementation before introducing any modifications in later phases of the project.

## Dataset
- File: `mxmh_survey_results.csv`
- Description: Survey-based dataset containing user demographics, music listening behavior, and self-reported mental health metrics (anxiety, depression, insomnia, OCD).

## Methodology
The implementation follows the approach described in the parent paper:

- Data preprocessing:
  - Missing value handling (imputation)
  - Outlier removal (age, BPM, listening hours)
- Feature types:
  - Numerical: age, hours_per_day, BPM
  - Categorical: streaming service, genre, listening habits, etc.
- Target variables:
  - Anxiety, Depression, Insomnia, OCD
  - Converted into binary classification (threshold-based)

## Models Implemented
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- Neural Network (MLP)

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

## Requirements
Install required libraries using:

```bash
pip install -r requirements.txt