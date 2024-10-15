"""
This module provides a Gradient Boosting Decision Tree (GBDT) trainer using LightGBM.
It includes methods for setting training and validation data, training the model,
making predictions, and saving/loading the model.

Classes:
    GBDT: A class for training and using a GBDT model with LightGBM.

Methods:
    __init__: Initializes the GBDT class.
    set_data: Sets the training and validation data for the model.
    train: Trains the GBDT model with the given parameters.
    predict: Makes predictions using the trained GBDT model.
    save: Saves the trained model to a file.
    load: Loads a trained model from a file.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

from typing import Any, List

import lightgbm as lgb
import numpy as np
import pandas as pd


class GBDT:
    def __init__(self):
        self.bst = None

    def set_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> None:
        """
        Sets the training and validation data for the GBDT model.

        Args:
            X_train (pd.DataFrame): Training feature data.
            y_train (pd.Series): Training target data.
            X_valid (pd.DataFrame): Validation feature data.
            y_valid (pd.Series): Validation target data.

        Returns:
            None
        """
        self.train_data = lgb.Dataset(
            data=X_train,
            label=y_train,
            feature_name="auto",
            params={"min_data_in_leaf": 10},
        )
        self.valid_data = lgb.Dataset(
            data=X_valid,
            label=y_valid,
            feature_name="auto",
            params={"min_data_in_leaf": 10},
        )

    def train(self, train_params: dict = {}) -> lgb.Booster:
        """
        Trains a LightGBM model using the provided training parameters.
        Args:
            train_params (dict): A dictionary of training parameters for LightGBM.
        Returns:
            lgb.Booster: The trained LightGBM Booster model.
        """

        bst = lgb.train(
            params=train_params,
            train_set=self.train_data,
            valid_sets=[self.valid_data],
            callbacks=[lgb.log_evaluation()],
        )
        self.bst = bst

        return bst

    def predict(self, X: pd.DataFrame) -> np.ndarray[Any, Any] | Any | List[Any]:
        """
        Makes predictions using the trained GBDT model.

        Args:
            X (pd.DataFrame): Feature data for making predictions.

        Returns:
            np.ndarray: Predicted values.
        """
        if X is not None:
            ypred = self.bst.predict(X)
            return ypred
        else:
            raise ValueError(f"X is None: {X}")

    def save(self, file_path: str) -> None:
        """
        Saves the trained model to a file.

        Args:
            file_path (str): The path to the file where the model will be saved.

        Returns:
            None
        """
        self.bst.save_model(file_path)

    def load(self, file_path: str) -> None:
        """
        Loads a trained model from a file.

        Args:
            file_path (str): The path to the file from which the model will be loaded.

        Returns:
            None
        """
        self.bst = lgb.Booster(model_file=file_path)
