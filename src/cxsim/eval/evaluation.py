"""
This module provides a utility class `Evaluation` for calculating various evaluation metrics
for binary classification and regression tasks. It includes methods for calculating metrics
such as MSE, RMSE, MAE, MAPE, SMAPE, F1 Score, AUC, and PR AUC.

Classes:
    Evaluation: A class containing static methods to calculate different evaluation metrics
                and class methods to evaluate binary classification and regression models.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_curve,
    roc_curve,
)


class Evaluation:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def calc_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float | Any:
        return mean_squared_error(y_true, y_pred, squared=True)

    @staticmethod
    def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float | Any:
        return mean_squared_error(y_true, y_pred, squared=False)

    @staticmethod
    def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float | Any:
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calc_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | Any:
        return mean_absolute_percentage_error(y_true, y_pred) * 100

    @staticmethod
    def calc_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float | Any:
        score_list = []
        for t, p in zip(y_true, y_pred):
            # cannot divide by zero, 0/0
            if t == 0 and p == 0:
                score_list.append(0.0)
            else:
                score = np.abs(p - t) / (np.abs(p) + np.abs(t)) / 2
                score_list.append(score)
        smape = np.mean(score_list) * 100
        return smape

    @staticmethod
    def calc_f1_score(
        y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float | Any:
        return f1_score(y_true, y_pred, average=average)

    @staticmethod
    def calc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | Any:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_value = auc(fpr, tpr)
        return auc_value

    @staticmethod
    def calc_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | Any:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        return pr_auc

    @classmethod
    def evaluate_binary(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        metrics = {
            "f1_score": cls.calc_f1_score(y_true, y_pred),
            "auc": cls.calc_auc(y_true, y_prob),
            "pr_auc": cls.calc_pr_auc(y_true, y_prob),
        }
        return metrics

    @classmethod
    def evaluate_regression(
        cls, y_true: np.ndarray, y_pred: np.ndarray, skip_zero: bool = True
    ) -> Dict[str, float]:
        if skip_zero:
            y_pred_nonzero = y_pred[y_true > 0]
            y_true_nonzero = y_true[y_true > 0]
            mape = cls.calc_mape(y_true_nonzero, y_pred_nonzero)
        else:
            mape = cls.calc_mape(y_true, y_pred)
        metrics = {
            "mse": cls.calc_mse(y_true, y_pred),
            "rmse": cls.calc_rmse(y_true, y_pred),
            "mae": cls.calc_mae(y_true, y_pred),
            "mape": mape,
            "smape": cls.calc_smape(y_true, y_pred),
        }
        return metrics
