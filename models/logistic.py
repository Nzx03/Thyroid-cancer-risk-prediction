from sklearn import linear_model
import joblib
import pandas as pd

X_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_train.csv")
X_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_test.csv")
y_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_train.csv")
y_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_test.csv")

def train_logistic_model(X_train, y_train):

 model=linear_model.LogisticRegression(max_iter=1000)
 model.fit(X_train,y_train)
 return model

def predict_logistic_model(model,X_test):
   return model.predict(X_test)

def save_model(model, path="Thyroid-cancer-risk-prediction/models/logistic_model.pkl"):
   joblib.dump(model,path)

def load_model(path="Thyroid-cancer-risk-prediction/models/logistic_model.pkl"):
  return joblib.load(path)