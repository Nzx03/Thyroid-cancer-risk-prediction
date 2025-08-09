from xgboost import XGBClassifier
import pandas as pd
import joblib

X_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_train.csv")
X_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\X_test.csv")
y_train = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_train.csv").values.ravel()
y_test = pd.read_csv(r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\data\tabular\processed_tabular\y_test.csv").values.ravel()
def train_xg_model(X_train, y_train):
   
      model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,   
        objective='multi:softmax',  
        num_class=3, 
        random_state=42,
        eval_metric='mlogloss'
    )
      model.fit(X_train,y_train)
      return model

def predict_xg_model(model,X_test):
   return model.predict(X_test)

def save_model(model, path=r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\models\xgboost.pkl"):
   joblib.dump(model,path)

def load_model(path=r"C:\Users\Nazneen\ML project\Thyroid-cancer-risk-prediction\models\xgboost.pkl"):
  return joblib.load(path)  
