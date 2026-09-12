import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def calculate_evaluation_metrics(y_true, y_pred_prob, threshold=0.3):
    """
    Calculates comprehensive classification metrics emphasizing planet recall.
    """
    y_pred = (y_pred_prob >= threshold).astype(int).flatten()
    y_true = np.array(y_true).flatten()
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / len(y_true)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = float(auc(fpr, tpr))
    
    report_dict = classification_report(y_true, y_pred, target_names=['No Planet', 'Planet'], output_dict=True)
    
    return {
        "accuracy": round(float(accuracy), 4),
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "f1_score": round(float(f1), 4),
        "auc_score": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(threshold),
        "classification_report": report_dict
    }


def optimize_decision_threshold(y_true, y_pred_prob, target_recall=1.0):
    """
    Finds the highest decision threshold that satisfies target planet recall (e.g. 100% recall).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    
    # Find thresholds matching target recall
    valid_indices = np.where(tpr >= target_recall)[0]
    if len(valid_indices) > 0:
        # Choose highest threshold among valid
        optimal_idx = valid_indices[np.argmax(thresholds[valid_indices])]
        best_threshold = float(thresholds[optimal_idx])
    else:
        best_threshold = 0.3
        
    return best_threshold
