from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

X_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_train.csv")
X_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_test.csv")
y_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_train.csv").values.ravel()
y_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_test.csv").values.ravel()
def train_rf_model(X_train, y_train):
    model=RandomForestClassifier(n_estimators=200,max_depth=None,random_state=42,class_weight="balanced")
    model.fit(X_train,y_train)
    return model

def predict_randomforest_model(model,X_test):
   return model.predict(X_test)

def save_model(model, path=r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\models\random_forest.pkl"):
   joblib.dump(model,path)

def load_model(path=r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\models\random_forest.pkl"):
  return joblib.load(path)  
