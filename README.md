# Thyroid Cancer Risk Prediction

This project leverages XGBoost to predict thyroid disorders using key medical features, aiding in early diagnosis and improving patient outcomes.

## Features
- **Data Preprocessing**  
  - Handled missing values, outliers, and performed feature engineering.
  - Encoded categorical features and normalized numerical features.
  - Applied train-test split to generate **X_train, X_test, y_train, y_test** datasets.

- **Modeling**
  - Implemented **Random Forest** with `class_weight='balanced'` to handle class imbalance.
  - Implemented **XGBoost** with hyperparameter tuning for improved performance.
  - Applied **SMOTE** oversampling to address dataset imbalance.

- **Evaluation**
  - Compared model performance using accuracy, precision, recall, and F1-score.
  - Generated confusion matrices to analyze classification errors.
  - Visualized feature importance for better interpretability.

## Instructions
1. Clone the repository.
2. Install dependencies using:
```
pip install -r requirements.txt
```
3. Run the Jupyter Notebook to execute the entire workflow, from data preparation to model training and evaluation.

## Tech Stack
- **Languages:** Python 3.11
- **Libraries:** pandas, scikit-learn, imbalanced-learn, xgboost, joblib, matplotlib, seaborn
- **Models:** Random Forest, XGBoost, Logistic regression

## Results
- Achieved **85% accuracy** with improved recall for identifying thyroid-positive cases.
- Visualizations included for data distribution, feature importance, and model evaluation metrics.



## Instructions to Run
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd Thyroid-cancer-risk-prediction



## Install Dependencies
```bash
pip install -r requirements.txt