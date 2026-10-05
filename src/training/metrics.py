import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def expected_calibration_error(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError('Nonempty, finite aligned labels/probabilities required')
    if np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be in [0, 1]')
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.mean(idx == b) * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if np.any(idx == b)))


def classification_metrics(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p)), 'ece_10_bins': expected_calibration_error(y, p)}


def retrieval_metrics(ranked, relevant, k=5):
    if not relevant or k < 1 or len(ranked) != len(set(ranked)):
        raise ValueError('Nonempty relevance judgments, positive k, unique ranked IDs required')
    hits = [int(x in relevant) for x in ranked[:k]]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return {'precision': sum(hits)/k, 'recall': sum(hits)/len(relevant), 'ndcg': float(dcg/ideal)}
