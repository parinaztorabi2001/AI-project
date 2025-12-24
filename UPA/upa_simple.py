# upa_simple.py
# نسخه ساده‌تر پروژه UPA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import interpolate

print("=" * 80)
print("UPA PROJECT - SIMPLE VERSION")
print("=" * 80)

# Step 1: Load dataset
try:
    df = pd.read_csv('data.csv')
    print("✓ Dataset loaded successfully")
    print(f"Shape: {df.shape}")
except:
    print("✗ ERROR: data.csv not found!")
    print("Please ensure data.csv is in the same folder")
    exit()

# Step 2: Preprocess
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
le = LabelEncoder()
y = le.fit_transform(df['diagnosis'])  # B=0, M=1
X = df.drop('diagnosis', axis=1)

print(f"\nClasses: {list(le.classes_)} -> {list(le.transform(le.classes_))}")
print(f"Benign: {(y == 0).sum()}, Malignant: {(y == 1).sum()}")

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = (tp + tn) / (tp + tn + fp + fn)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 80)
print("BASIC RESULTS")
print("=" * 80)
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}")
print(f"Confusion Matrix:")
print(f"  TN: {tn}, FP: {fp}")
print(f"  FN: {fn}, TP: {tp}")


# Simple UPA implementation
class SimpleUPA:
    def __init__(self, n_bins=50):
        self.n_bins = n_bins
        self.mapping = None

    def fit(self, ref_preds, target_preds):
        ref_hist, ref_bins = np.histogram(ref_preds, bins=self.n_bins, range=(0, 1))
        target_hist, target_bins = np.histogram(target_preds, bins=self.n_bins, range=(0, 1))

        ref_cdf = np.cumsum(ref_hist) / np.sum(ref_hist)
        target_cdf = np.cumsum(target_hist) / np.sum(target_hist)

        self.mapping = np.interp(target_cdf, ref_cdf, ref_bins[:-1])
        self.target_bins = target_bins

    def transform(self, preds):
        if self.mapping is None:
            return preds

        aligned = np.zeros_like(preds)
        for i, p in enumerate(preds):
            idx = np.digitize(p, self.target_bins) - 1
            idx = np.clip(idx, 0, len(self.mapping) - 1)
            aligned[i] = self.mapping[idx]

        return np.clip(aligned, 0, 1)


# Create simple visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Prediction distribution
axes[0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Benign', density=True)
axes[0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Malignant', density=True)
axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
axes[0].set_xlabel('Prediction Probability')
axes[0].set_ylabel('Density')
axes[0].set_title('Breast Cancer Classification')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Performance metrics
metrics = ['Sensitivity', 'Specificity', 'Accuracy', 'AUC']
values = [sensitivity, specificity, accuracy, auc]
colors = ['blue', 'green', 'orange', 'red']

bars = axes[1].bar(metrics, values, color=colors, alpha=0.7)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('Score')
axes[1].set_title('Model Performance Metrics')
axes[1].grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('upa_simple_results.png', dpi=300, bbox_inches='tight')

print("\n" + "=" * 80)
print("✓ Project completed!")
print("✓ Results saved to 'upa_simple_results.png'")
print("=" * 80)

plt.show()