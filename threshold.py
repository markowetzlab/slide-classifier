import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('/media/prew01/BEST/BEST4/pilot/best4_pilot.csv')
csv_path = '/media/prew01/BEST/BEST4/pilot/p53/40x_400/inference'
csvs = os.listdir(csv_path)

files = df['Cyted Sample Instance ID']
# ground_truth = df['Post-Cutting Atypia'].map({'N': 0, 'To Clarify':0, 'Y': 1})
ground_truth = df['Post-Cutting p53'].map({'N': 0, 'EQV':0, 'WT':1, 'Y': 1})

thresholds = np.append(np.arange(0, 0.99, 0.01), np.arange(0.99, 0.999, 0.001))
# thresholds = np.append(thresholds, np.arange(0.9999, 0.99999, 0.00001))

file_results = []

for file in files:
    csv_file = [c for c in csvs if c.startswith(file)]
    if not csv_file:
        continue
    positive_tiles = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        model_outputs = pd.read_csv(os.path.join(csv_path, csv_file[0]))
        # target_class = model_outputs['atypia']
        target_class = model_outputs['aberrant_positive_columnar']
        positive_tiles[i] += ((target_class >= threshold)).sum()
    file_results.append(positive_tiles)

results = pd.DataFrame(file_results)
column_names = [f"Threshold_{i}" for i in thresholds]
results.columns = column_names

# Initialize accumulators for TP, TN, FP, FN across all files for each threshold
total_tp, total_tn, total_fp, total_fn = np.zeros(len(thresholds)), np.zeros(len(thresholds)), np.zeros(len(thresholds)), np.zeros(len(thresholds))

# Initialize arrays to store sensitivity and specificity for each threshold
sensitivity_scores = np.zeros(len(thresholds))
specificity_scores = np.zeros(len(thresholds))

target_sensitivity = 0.7
best_threshold = None
max_specificity = 0
sensitivity_target = 0

# Calculate specificity and sensitivity for each threshold after accumulating results
for i, threshold in enumerate(thresholds):

    predictions = (results[f'Threshold_{threshold}'] >= 1).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    # Check if this threshold's specificity is equal to or greater than the target specificity
    # and if its sensitivity is the highest found so far
    if sensitivity >= target_sensitivity and specificity > max_specificity:
        max_specificity = specificity
        sensitivity_target = sensitivity
        best_threshold = threshold

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity

# Output the best threshold and its sensitivity
print(f"Best Threshold: {best_threshold}, Maximum Specificity: {max_specificity} at {sensitivity_target}")

# Calculate Youden's J statistic for each threshold
youden_j = sensitivity_scores + specificity_scores - 1

# Find the index of the maximum Youden's J statistic
optimal_index = np.argmax(youden_j)

# Find the optimal threshold
optimal_threshold = thresholds[optimal_index]

print(f'Optimal Threshold: {optimal_threshold}')
print(f'Maximizes Sensitivity: {sensitivity_scores[optimal_index]}')
print(f'Maximizes Specificity: {specificity_scores[optimal_index]}')

# Calculate FPR from specificity scores
fpr = 1 - specificity_scores

# Sort FPR and corresponding TPR (sensitivity)
sorted_indices = np.argsort(fpr)
sorted_fpr = fpr[sorted_indices]
sorted_tpr = sensitivity_scores[sorted_indices]

# Calculate AUROC
auroc = auc(sorted_fpr, sorted_tpr)

# Print AUROC
print(f'AUROC: {auroc}')

# Plot ROC curve
plt.figure(figsize=(10, 5))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(sorted_fpr, sorted_tpr, label='ROC curve (area = %0.2f)' % auroc)
ax1.plot([0, 1], [0, 1], 'k--')  # Random chance line
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic')
ax1.legend(loc="lower right")

# Plot Sensitivity and Specificity vs. Thresholds
ax2 = plt.subplot(1, 2, 2)
ax2.plot(thresholds, sensitivity_scores, label='Sensitivity', color='blue')
ax2.plot(thresholds, specificity_scores, label='Specificity', color='red')  # 1 - FPR is specificity
ax2.set_xlabel('Threshold')
ax2.set_ylabel('Score')
ax2.set_title('Sensitivity and Specificity vs. Threshold')
ax2.legend(loc="best")

plt.tight_layout()
plt.show()