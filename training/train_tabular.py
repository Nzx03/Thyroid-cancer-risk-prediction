from models.logistic import train_logistic_model, predict_logistic_model
from evaluations.evaluate_tabular import evalaute_model
import pandas as pd
   
from models.logistic import save_model
import sys
import os
sys.path.append(os.path.abspath('.'))  # Add project root to sys.path

def load_data():
    X_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_train.csv")
    X_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_test.csv")
    y_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_train.csv")
    y_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_test.csv")
    return X_train, X_test, y_train, y_test

def main():
   X_train, X_test, y_train, y_test=load_data()
   model=train_logistic_model(X_train,y_train)
   predict=predict_logistic_model(model, X_test)
   evalaute_model(y_test, predict)
   save_model(model)

if __name__=="__main__":
    main()          #for preventing automatic execution
