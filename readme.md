# Home Equity Line of Credit Default Prediction

## Overview:

In this project, our objective is to predict whether a potential borrower will default on their home equity line of credit (HELOC) based on historical data and credit-related attributes. We will utilize various machine learning classifiers to build predictive models and compare their performance.

The banking industry bases its Expected Loss of its decisions based on three factors. Probability of default (PD), loss given default (LGD) and exposure at default (EG). The product of these three is the banks expected loss for the given credit program. In our case, we are going to determine a models ability to predict whether a default will occur or not. 

The dataset contains information related to the borrower's credit history, external risk estimates, and various other credit-related indicators. The target variable for prediction is the "RiskPerformance" feature, indicating whether the borrower has a good or bad credit performance (binary: 'good' or 'bad'). We will use this as the target to train our machine learning model. The machine will turn around and tell us how accurate it can be in predicting this. 

Our goal is to identify the best-performing model that can precisely predict default risk for potential borrowers in a trustworthy manner.

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

## Data Deeper Dive

The features we are selecting are as follows as their correlate most with our target of Risk Performance. These are:

NetFractionRevolvingBurden            0.298174
ExternalRiskEstimate                  0.216770
AverageMInFile                        0.209168
PercentTradesWBalance                 0.198554
MSinceOldestTradeOpen                 0.185155
NumSatisfactoryTrades                 0.123080
PercentTradesNeverDelq                0.122010
PercentInstallTrades                  0.111542
MSinceMostRecentInqexcl7days          0.110253
MaxDelq2PublicRecLast12M              0.109946
MaxDelqEver                           0.107204

The reasons we are chosing these, as you can see there is a value next to them. This value is the correlation amount these features have with the output of Risk Performance. Now none of these have an extraordinary high correlation so it would be hard for a human to determine a prediction based on this information. However a combination of these factors in a non-linear fashion can be done by a computer, and that is what we will do. We will use these features, to create a model based on training data from our dataset and we will evaluate those models based on these factors in the next section.

## Evalutation Metrics

Metrics: Accuracy, Precision, Recall, F1 Score, Area Under the Curve (AUC) and overfitting.

I will explain why these metrics matter and how it plays a part in our credit industry. Accuracy is the overall accuracy of the model, as this is helpful metric, it isn't the most important metric in our case because this tells us how often the model correctly predicts a Good borrower who truly is a good borrower. As this is important, we can be harmed much worse if we falsely predict a bad borrower as a good borrower, therefore enter precision score. Precision will tell us how good we are prevent false positives ie. predicting a truly bad borrower is a good one. This will reduce our expected loss the most in this case, so we are going to primarly focus on this metric while closely watching the others. Recall is a metric that will tell us how well we are identifying all good borrowers to maximize our profits. This is important because we are in the business of making money, however our losses due to default usually out weight heavily on a monetary scale the amount of damage we incurr opposed to the interest we make from someone. If we have to, we will side with a model that has better precision.

Now, there is an interesting metric I want to share with you, the F1 score. This is a blend of precision AND recall. Therefore, this is a good overall score to be aware of as how the balance between precision and recall is doing. Another metric we will look at is Area Under the Curve. This helps us understand how good our model is. The AUC score tells us how well our model separates good and bad answers. A score above .5 is good here because .5 means the model is strictly guessing.

Lastly, we have a self created metric call overfitting. Overfitting is when a model follows the real world data too closely, it is very bad when it comes in contact with real world data. Meaning, it created a model based on its sample given to train, opposed to being good at utilization in real world. We use test data set that the model hasnt ever seen before to see how well it is getting trained to be used in the real world. The difference between Train and Test scores should not be over .05 in my estimation. If there is a bigger gap than .05 then we may have some cause for concern that our model is overfitting the training data we gave it.

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

We really like XGBoost for the reasons provided in the evaluation metrics explanations and the scores we see above.

We also believe there is great potential to explain this model with the use of shapley values/rates. It makes us confident that the scores we are seeing are similar to what won the FICO competition for this is same dataset.

## Recommendations:

Considering the performance and overfitting analysis, we recommend proceeding with the following models for predicting default risk in the Home Equity Line of Credit application in ordered by highest test precision score.:

1. XGBoost


These models have demonstrated acceptable performance and do not suffer from overfitting, however further tuning of the models parameters can make this go south very fast. It is very important we know what we are doing before adjusting these models. In addition, we must closely watch any changes from the real world data.  There may be an additional tradeoff we will encounter in the next exercise in terms of interpretability or explainability of these models. The business use case also exists in highly regulated industry so every decision will have to be auditable and explainable.

## Suggestions for Next Steps

We should deploy this model against the real world data into production. While running simoustaneous tests of these other backup models. If we start seeing differences in test scores in the real world data from our train/test metrics, then we know we have a cause for concern. This will require constant monitoring of our models and adjustments in real time so that our decisions are not done in inappropriate fashion. I will keep you in the loop of any more developments. 

## Notebook Link:

For a detailed view of the entire project, including data exploration, preprocessing, model training, evaluation, and hyperparameter tuning, please refer to the Jupyter Notebook available at [View Capstone_McGuire_HELOC_Final.ipynb)](https://github.com/vmcguire/capstone_final/Capstone_McGuire_HELOC_Final.ipynb)