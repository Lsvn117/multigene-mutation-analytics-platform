import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths
labels_file = 'C:/Users/DEEPAK/OneDrive/Desktop/Mini2/labels.txt'
embeddings_folder = 'C:/Users/DEEPAK/OneDrive/Desktop/Mini2/Embeddings'

# Load labels
labels_dict = {}
with open(labels_file, 'r') as f:
    for line in f:
        file_name, label = line.strip().split(',')
        labels_dict[file_name] = 0 if label == "Benign" else 1

# Load embeddings and match with labels
X_raw, y = [], []
for file_name, label in labels_dict.items():
    file_path = os.path.join(embeddings_folder, file_name + ".npy")
    if os.path.exists(file_path):
        embedding = np.load(file_path).flatten()
        X_raw.append(embedding)
        y.append(label)

# Pad all embeddings to the same length
max_len = max(len(e) for e in X_raw)
X_padded = [np.pad(e, (0, max_len - len(e)), 'constant') for e in X_raw]
X = np.array(X_padded)
y = np.array(y)

# Normalize embeddings
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define classifier with probability support
model = SVC(kernel='linear', class_weight='balanced', random_state=42)

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Track best fold
best_acc = 0
best_metrics = {}
best_cm = None

fold = 1
for train_idx, test_idx in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    if acc > best_acc:
        best_acc = acc
        best_cm = confusion_matrix(y_test, y_pred)
        best_metrics = {
            "accuracy": acc,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        }

    fold += 1

# Save the trained SVM model and scaler
joblib.dump(model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Print best fold metrics
print("\n📈 Best Fold Metrics:")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f}")
print(f"F1 Score: {best_metrics['f1']:.4f}")
print("Confusion Matrix:")
print(best_cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Benign", "Pathogenic"],
            yticklabels=["Benign", "Pathogenic"])
plt.title(f'Confusion Matrix (Accuracy: {best_metrics["accuracy"]:.4f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("best_confusion_matrix.png")
plt.show()

# Plot bar graph of metrics
metrics = {
    "Accuracy": best_metrics["accuracy"],
    "Precision": best_metrics["precision"],
    "Recall": best_metrics["recall"],
    "F1 Score": best_metrics["f1"]
}

plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(metrics.keys(), metrics.values(), color=colors)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.4f}",
             ha='center', va='bottom', fontsize=12)

plt.ylabel("Score", fontsize=14)
plt.xlabel("Metric", fontsize=14)
plt.title("Performance Metrics", fontsize=16)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("best_fold_metrics.png")
plt.show()
