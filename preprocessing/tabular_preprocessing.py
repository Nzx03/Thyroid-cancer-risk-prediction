import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

def preprocess_tabular(input_data, output_data):

     data = pd.read_csv(input_data)
     #using raw strings to read data
     data.shape
     data.info()
     Q1 = data['TSH_Level'].quantile(0.25)
     Q3 = data['TSH_Level'].quantile(0.75)
     IQR = Q3 - Q1
     filtered_data = data[(data['TSH_Level'] >= Q1 - 1.5 * IQR) & 
                     (data['TSH_Level'] <= Q3 + 1.5 * IQR)].copy()  # `.copy()` ensures a fresh copy



    #  print(data.shape)  # To confirm no columns are removed
    #  print(data.head()) #removing outlier using IQR method 
      
     #Assuming 'data' is your original DataFrame
     data['Diagnosis_Encoded'] = (data['Diagnosis'] == 'Malignant').astype('int8')

    # Drop the original 'Diagnosis' column
     data.drop('Diagnosis', axis=1, inplace=True)

    #  print(data.head())
    #  print(data['Diagnosis_Encoded'].value_counts())# To verify the conversion
    # To verify the conversion
     le = LabelEncoder()
     data['Gender'] = le.fit_transform(data['Gender'])
     data = pd.get_dummies(data, columns=['Country', 'Ethnicity'])

    #  Sample data structure (for reference)
    # Assuming 'data' is your original DataFrame with 15+ columns
    # 'Thyroid_Risk' is one of the columns

     encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
     data['Thyroid__Cancer_Risk_Encoded'] = encoder.fit_transform(data[['Thyroid_Cancer_Risk']]).astype(int)

    # Drop the original 'Thyroid_Risk' column
     data.drop('Thyroid_Cancer_Risk', axis=1, inplace=True)

    #  print(data.head())
     scaler = StandardScaler()
     scaled_features = scaler.fit_transform(data[['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']])
     data[['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']] = scaled_features
     data['Family_History'] = data['Family_History'].map({'No': 0, 'Yes': 1}).astype('int8')
     data['Radiation_Exposure'] = data['Radiation_Exposure'].map({'No': 0, 'Yes': 1}).astype('int8')
     data['Iodine_Deficiency'] = data['Iodine_Deficiency'].map({'No': 0, 'Yes': 1}).astype('int8')
     data['Smoking'] = data['Smoking'].map({'No': 0, 'Yes': 1}).astype('int8')
     data['Obesity'] = data['Obesity'].map({'No': 0, 'Yes': 1}).astype('int8')
     data['Diabetes'] = data['Diabetes'].map({'No': 0, 'Yes': 1}).astype('int8')

    #  print(data['Family_History'].value_counts())  # To verify the conversion
    #  print(data['Radiation_Exposure'].value_counts())  # To verify the conversion
    #  print(data['Iodine_Deficiency'].value_counts())  # To verify the conversion
    #  print(data['Smoking'].value_counts())  # To verify the conversion
    #  print(data['Obesity'].value_counts())  # To verify the conversion
    #  print(data['Diabetes'].value_counts())  # To verify the conversion
    #  print(data.head())
    
     X = data.drop("Thyroid__Cancer_Risk_Encoded", axis=1)
     y = data["Thyroid__Cancer_Risk_Encoded"]

    # Train-test split
     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

     os.makedirs(output_data, exist_ok=True)

    
     X_train.to_csv(os.path.join(output_data, "X_train.csv"), index=False)
     X_test.to_csv(os.path.join(output_data, "X_test.csv"), index=False)
     y_train.to_csv(os.path.join(output_data, "y_train.csv"), index=False)
     y_test.to_csv(os.path.join(output_data, "y_test.csv"), index=False)

     print("files saved sucessfully in",output_data)

