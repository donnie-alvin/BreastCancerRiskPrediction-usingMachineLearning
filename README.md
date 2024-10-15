# BreastCancerRiskPrediction-usingMachineLearning

This project demonstrates the implementation of machine learning models to predict breast cancer risk using two publicly available datasets: the **Wisconsin Diagnostic Breast Cancer (WDBC) dataset** and the **Breast Cancer Coimbra dataset**. The models used in this study include Logistic Regression, Random Forest, and Support Vector Machines (SVM), all implemented using the **scikit-learn** library.

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)

## Introduction
The goal of this project is to evaluate machine learning models' effectiveness in predicting breast cancer risk. The models are trained on breast cancer datasets to classify the cancer diagnosis as either benign or malignant. This repository contains the implementation of the models and a detailed comparison of their performance.

## Datasets
One dataset is used in this project:

1. **Breast Cancer Coimbra dataset**: A dataset containing clinical features for breast cancer prediction.

You can download the dataset from:
- [Breast Cancer Coimbra dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra)

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/donnie-alvin/BreastCancerRiskPrediction-usingMachineLearning.git
   cd breast-cancer-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the models by executing the `main.py` file:
```bash
python main.py
```

The results will be printed in the terminal, displaying the performance metrics (accuracy, precision, recall, F1-score) for each model.

## Machine Learning Models
The following machine learning models were implemented using the **scikit-learn** library:
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)

Each model was trained on the datasets and evaluated using cross-validation techniques. The hyperparameters were adjusted using grid search to improve performance.

## Evaluation
The models are evaluated using the following performance metrics:
- **Accuracy**: The overall correctness of the model.
- **Precision**: The proportion of true positive results among the positive predictions.
- **Recall**: The proportion of true positive results out of all actual positives.
- **F1-score**: The harmonic mean of precision and recall.
- **AUC-ROC**: The area under the ROC curve, which measures the trade-off between true positive rate and false positive rate.

## Results
The following results were obtained for each model (with default parameters):

| Model               | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 87.5%    | 0.88      | 0.88   | 0.88     | 0.88    |
| Random Forest       | 83.33%   | 0.84      | 0.83   | 0.83     | 0.83    |
| SVM                 | 79.17%   | 0.81      | 0.79   | 0.79     | 0.79    |

## Future Work
- **Fine-tuning of models**: Further parameter tuning can be done to enhance model performance.
- **Exploring more algorithms**: Models such as deep learning networks can be explored.
- **Feature Engineering**: Additional feature selection techniques can be implemented to improve prediction accuracy.

## License
