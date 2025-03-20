# LLMs-Can-t-Please-Them-All
# README

This repository contains code for training various machine learning models to predict tournament results using logistic regression, random forest, and gradient boosting techniques. The implementation supports both men's and women's tournaments.

## Model Training

### Logistic Regression

#### Steps:
1. Mount Google Drive to access datasets.
2. Load and preprocess data.
3. Train a logistic regression model using `LogisticRegression` from `sklearn`.
4. Calibrate the model using `CalibratedClassifierCV`.
5. Generate predictions for tournament teams.
6. Save results in a submission file.

#### Code Overview:
- Loads data from Google Drive.
- Performs feature engineering (calculates score difference as a feature).
- Splits data into training and testing sets.
- Uses a pipeline with standardization and logistic regression.
- Calibrates the model for better probability estimates.
- Generates win probability predictions for each team.
- Creates a CSV submission file for tournament predictions.

### Random Forest

#### Steps:
1. Load preprocessed features from logistic regression section.
2. Train a `RandomForestClassifier`.
3. Calibrate using `CalibratedClassifierCV`.
4. Predict tournament outcomes.

#### Issues:
- `NameError: name 'X' is not defined` missing feature definitions.

### Gradient Boosting Machines (LightGBM)

#### Steps:
1. Define a `LGBMClassifier` model.
2. Calibrate with `CalibratedClassifierCV`.
3. Train and predict probabilities for tournament teams.

#### Issues:
- `NameError: name 'X_train' is not defined` due to missing variable assignment.

### Grid Search CV (LightGBM)

#### Steps:
1. Define a parameter grid for hyperparameter tuning.
2. Use `GridSearchCV` with `brier_score_loss` to optimize model performance.

## Submission File Generation

A function `create_submission_file_2025()` generates a CSV file containing predictions for matchups in the 2025 NCAA tournament. It combines results from men's and women's tournaments.

