# StartupShield AI Viva Questions and Answers

## What is the problem solved by this project?

StartupShield AI estimates whether a startup is likely to be acquired or closed using historical startup information. It is a binary classification problem.

## What is the target variable?

`labels` is the target variable: `1` represents Acquired and `0` represents Closed. The text column `status` represents the same outcome, so it is excluded from model inputs.

## Why was `closed_at` removed?

It can reveal the eventual outcome of a startup. Using it to predict closure would be data leakage, because that information would not be available at an earlier prediction time.

## Why use a train/test split?

The model learns from the training data. The untouched test data works as a final exam to estimate performance on unseen startups.

## Why use `stratify=target`?

It preserves the acquired/closed class ratio in both the training and test datasets.

## Why compare three models?

Logistic Regression provides a linear baseline, Decision Tree captures rule-like patterns, and Random Forest combines many trees to capture more complex patterns.

## Why choose F1-score for selection?

F1-score balances precision and recall. It is more informative than accuracy alone when class distributions are not perfectly balanced.

## What is GridSearchCV?

GridSearchCV tests a defined set of hyperparameter combinations using cross-validation and selects the configuration with the best validation score.

## What is SHAP?

SHAP is an explainability method. It estimates how much each feature pushes a particular prediction toward Acquired or Closed.

## What are project limitations?

The system uses historical data, identifies associations rather than causation, and does not provide investment advice or guarantee future startup outcomes.
