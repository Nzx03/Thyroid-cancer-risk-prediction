#Calculating accuracy, precision, recall, F1 
from sklearn.metrics import accuracy_score, confusion_matrix

def evalaute_model(y_test,y_pred):
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Accuracy;{accuracy:.4f}")

    matrix=confusion_matrix(y_test,y_pred)

    print(f"Confuison Matrix:{matrix}")

    return{
        "accuracy": accuracy,
        "confusion_matrix":matrix
    }