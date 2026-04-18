import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Loading evaluation module...")

try:
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report
    )

    SKLEARN_AVAILABLE = True

except Exception as e:
    print("Warning: sklearn not working properly")
    print(e)
    SKLEARN_AVAILABLE = False


# ============================================
# BASIC METRICS
# ============================================

def evaluate_model(y_true, y_pred):
    if not SKLEARN_AVAILABLE:
        print("Cannot evaluate model — sklearn not working")
        return

    print("\nMODEL EVALUATION")
    print("=" * 40)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


# ============================================
# CONFUSION MATRIX
# ============================================

def plot_confusion_matrix(y_true, y_pred, labels):
    if not SKLEARN_AVAILABLE:
        print("Cannot plot confusion matrix — sklearn not working")
        return

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# ============================================
# SIMPLE TEST
# ============================================

if __name__ == "__main__":
    print("Evaluation module loaded (safe mode).")
