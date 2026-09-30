import json

print('=' * 50)
print('XGBOOST METRICS')
print('=' * 50)

with open('models/metrics.json', 'r') as f:
    m = json.load(f)
    x = m['xgboost']

print(f'Accuracy: {x["accuracy"]:.2%}')
print(f'Balanced Accuracy: {x["balanced_accuracy"]:.2%}')
print(f'AUC-ROC: {x["auc_roc"]:.4f}')  # ← Changed from 'auc' to 'auc_roc'
print(f'Precision: {x["precision"]:.4f}')
print(f'Recall: {x["recall"]:.4f}')
print(f'F1 Score: {x["f1"]:.4f}')
print(f'Optimal Threshold: {x.get("best_threshold", 0.5):.3f}')

print('\n' + '=' * 50)
print('LSTM METRICS')
print('=' * 50)

with open('models/metrics.json', 'r') as f:
    l = m['lstm']  # LSTM is in the same file

print(f'Accuracy: {l["accuracy"]:.2%}')
print(f'Balanced Accuracy: {l["balanced_accuracy"]:.2%}')
print(f'AUC-ROC: {l["auc_roc"]:.4f}')
print(f'Precision: {l["precision"]:.4f}')
print(f'Recall: {l["recall"]:.4f}')
print(f'F1 Score: {l["f1"]:.4f}')
print(f'Optimal Threshold: {l.get("optimal_threshold", 0.5):.3f}')