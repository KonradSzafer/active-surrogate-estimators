"""Define losses."""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import balanced_accuracy_score


class SELoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target, *args, **kwargs):
        """Does not aggregate."""
        return (pred-target)**2


class MSELoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target, *args, **kwargs):
        """Aggregates to single digit."""
        return (SELoss()(pred, target, *args, **kwargs)).mean()


class RMSELoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target, *args, **kwargs):
        """Aggregates to single digit."""
        return np.sqrt(MSELoss()(pred, target, *args, **kwargs))


class AccuracyLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target):
        """Compute 1 - accuracy.

        Expects pred to be probabilities NxC and target to be in [1,..., C].

        Currently inconsistent with Crossentropy loss.

        """
        return 1. - (np.argmax(pred, axis=1) == target).astype(np.float)


class TPRLoss:
    def __call__(self, pred, target):
        pred = np.argmax(pred, axis=1)
        TP = 0
        FN = 0
        for i in range(len(target)):
            if target[i] == 1 and pred[i] == 1:
                TP += 1
            elif target[i] == 1 and pred[i] == 0:
                FN += 1
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        return TPR


class FPRLoss:
    def __call__(self, pred, target):
        pred = np.argmax(pred, axis=1)
        FP = 0
        TN = 0
        for i in range(len(target)):
            if target[i] == 0 and pred[i] == 1:
                FP += 1
            elif target[i] == 0 and pred[i] == 0:
                TN += 1
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        return FPR


class BalancedAccuracy:
    def __call__(self, pred, target):
        if pred.ndim == 2:
            pred = np.argmax(pred, axis=1)
        balanced_accuracy = balanced_accuracy_score(target, pred)
        return balanced_accuracy


class BalancedAccuracyLoss:
    def __call__(self, pred, target):
        if pred.ndim == 2:
            pred = np.argmax(pred, axis=1)
        balanced_accuracy = balanced_accuracy_score(target, pred)
        return 1 - balanced_accuracy


class CrossEntropyLoss:

    enc = None
    eps = 1e-19

    def __call__(self, pred, target):
        """Compute Cross-entropy loss.

        TODO: Numerical instabilities?
        pred: Predicted probabilities, NxC
        target: true class values in [1,..., C], N times
        """

        # One-Hot Encode
        if CrossEntropyLoss.enc is None:
            CrossEntropyLoss.enc = OneHotEncoder(sparse_output=False)
            CrossEntropyLoss.enc.fit(
                np.arange(0, pred.shape[1])[..., np.newaxis])

        # Clipping
        pred = np.clip(pred, self.eps, 1 - self.eps)
        # Renormalize
        pred /= pred.sum(axis=1)[:, np.newaxis]

        one_hot = CrossEntropyLoss.enc.transform(target[..., np.newaxis])
        res = -1 * (one_hot * np.log(pred)).sum(axis=1)

        return res


class YIsLoss:
    """Hijacking this code for toy experiment.

    For the toy experiment, we want the TrueRisk to be equal to the unknown
    f(x) = y.

    (We will then later compare TrueRiskEsimator to estimates from our model.)
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target, *args, **kwargs):
        """Does not aggregate."""
        return target
