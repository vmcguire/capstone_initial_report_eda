# Home Equity Line of Credit Default Prediction

## Overview:

In this project, our objective is to predict whether a potential borrower will default on their home equity line of credit (HELOC) based on historical data and credit-related attributes. We will utilize various machine learning classifiers to build predictive models and compare their performance.

The dataset contains information related to the borrower's credit history, external risk estimates, and various other credit-related indicators. The target variable for prediction is the "RiskPerformance" feature, indicating whether the borrower has a good or bad credit performance (binary: 'good' or 'bad').

Our goal is to identify the best-performing model that can accurately predict default risk for potential borrowers.

## Data:

The dataset contains a total of 41,188 records with 21 columns. The columns in the dataset include features like "ExternalRiskEstimate," "MSinceOldestTradeOpen," "NumSatisfactoryTrades," "PercentTradesNeverDelq," and many others. These features provide valuable insights into the borrower's credit history and repayment behavior.

The data used in this project comes from the Home Equity Line of Credit (HELOC) dataset available on Kaggle. You can find the dataset at the following link:

[Home Equity Line of Credit (HELOC) Dataset](https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc)

## Approach:

1. Data Exploration: We will start by exploring the dataset to gain an understanding of the feature distributions, check for missing values, and analyze the target variable's distribution.
2. Feature Selection: To focus on the most relevant features, we will consider features that have a correlation of 0.1 or higher with the target variable ("RiskPerformance").
3. Data Preprocessing: We will handle any missing values and perform data preprocessing tasks, such as encoding categorical variables and ensuring data consistency.
4. Model Selection: We will compare many different classifiers to include: XGBoost, AdaBoost, Logistic Regression, and Neural Networks.
5. Train-Test Split: We will divide the data into training and testing sets to evaluate the models' performance on unseen data.
6. Hyperparameter Tuning: We will use grid search with cross-validation to find the optimal hyperparameters for each model that performs best in the Model Selection phase.
7. Model Training and Evaluation: Each classifier will be trained on the training set, and their performance will be evaluated on the test set.
8. Overfitting Analysis: We will analyze each model for overfitting by comparing the difference in training and test accuracies, precision, recall, F1 score, and AUC.
9. Final assessment, we will be looking at all of the accuray, precision, recall, F1, AUC, and our customer built overfitting category to determine which models to prioritize and optimize later. Summary of our findings are below.

## Findings:

After evaluating the classifiers on the selected features, we obtained the following results:

1. XGBoost:
    - Train Accuracy: 0.7488, Test Accuracy: 0.7467
    - Train Precision: 0.7497, Test Precision: 0.7490
    - Train Recall: 0.7466, Test Recall: 0.7466
    - Train F1 Score: 0.7447, Test F1 Score: 0.7447
    - Train AUC: 0.8284, Test AUC: 0.8186
    - Overfitting Category: Acceptable
2. AdaBoost:
    - Train Accuracy: 0.7296, Test Accuracy: 0.7443
    - Train Precision: 0.7298, Test Precision: 0.7466
    - Train Recall: 0.7421, Test Recall: 0.7466
    - Train F1 Score: 0.7279, Test F1 Score: 0.7423
    - Train AUC: 0.8092, Test AUC: 0.8181
    - Overfitting Category: Acceptable
3. Logistic Regression:
    - Train Accuracy: 0.6967, Test Accuracy: 0.6659
    - Train Precision: 0.7036, Test Precision: 0.6786
    - Train Recall: 0.7245, Test Recall: 0.6792
    - Train F1 Score: 0.7139, Test F1 Score: 0.6789
    - Train AUC: 0.7549, Test AUC: 0.7229
    - Overfitting Category: Acceptable
4. Neural Network:
    - Train Accuracy: 0.6811, Test Accuracy: 0.6735
    - Train Precision: 0.6581, Test Precision: 0.6538
    - Train Recall: 0.8108, Test Recall: 0.7914
    - Train F1 Score: 0.7265, Test F1 Score: 0.7160
    - Train AUC: 0.7642, Test AUC: 0.7218
    - Overfitting Category: Acceptable

Based on the evaluation, all four models show acceptable or mild overfitting. The XGBoost model has the highest test precision, followed by AdaBoost, Logistic Regression, and Neural Network.

## Recommendations:

Considering the performance and overfitting analysis, we recommend proceeding with the following models for predicting default risk in the Home Equity Line of Credit application in ordered by highest test precision score.:

1. XGBoost
2. AdaBoost
3. Logistic Regression
4. Neural Network

These models have demonstrated acceptable performance and do not suffer from significant overfitting. However, it is essential to continue monitoring and updating the models with new data to maintain their predictive accuracy. There may be an additional tradeoff we will encounter in the next exercise in terms of interpretability or explainability of these models. The business use case also exists in highly regulated industry so every decision will have to be auditable and explainable.

## Notebook Link:

For a detailed view of the entire project, including data exploration, preprocessing, model training, evaluation, and hyperparameter tuning, please refer to the Jupyter Notebook available at [View Home Equity Line of Credit Default Prediction.ipynb](https://github.com/vmcguire/)