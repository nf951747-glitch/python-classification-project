from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Predicts on test data and prints accuracy, confusion matrix,
    and F1 score directly to the terminal — no plot window.
    """
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    cm = confusion_matrix(y_test, predictions)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))

    print_confusion_matrix(cm, class_names)

    return cm, acc, f1

def print_confusion_matrix(cm, class_names):
    """
    Prints the confusion matrix as a readable table in the terminal.
    Rows = actual class, Columns = predicted class.
    """
    print("Confusion Matrix:")
    header = "        " + "  ".join(f"{name[:10]:>10}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>10}" for val in row)
        print(f"{class_names[i][:8]:>8}  {row_str}")