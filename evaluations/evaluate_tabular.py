#Calculating accuracy, precision, recall, F1 


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{matrix}")

    precision = precision_score(y_test, y_pred, average=None)
    print("Precision:")
    for index, p in enumerate(precision):
        print(f"Class {index}: {p:.4f}")

    recall = recall_score(y_test, y_pred, average=None)
    print("Recall:")
    for index, r in enumerate(recall):
        print(f"Class {index}: {r:.4f}")

    f1 = f1_score(y_test, y_pred, average=None)
    print("F1 Score:")
    for index, f in enumerate(f1):
        print(f"Class {index}: {f:.4f}")

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
