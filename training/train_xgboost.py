import sys
import os
sys.path.append(os.path.abspath('.')) # Adding project root to sys.path
from models.xgboost_model import train_xg_model, predict_xg_model, save_model

from evaluations.evaluate_tabular import evaluate_model
import pandas as pd



def load_data():
    X_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_train.csv")
    X_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_test.csv")
    y_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_train.csv").values.ravel()
    y_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def main():
   X_train, X_test, y_train, y_test=load_data()
   model=train_xg_model(X_train,y_train)
   predict=predict_xg_model(model, X_test)
   metrics=evaluate_model(y_test, predict)
   print(metrics)
   save_model(model)

if __name__=="__main__":
    main()          #for preventing automatic execution
