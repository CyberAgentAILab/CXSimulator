"""
This module contains the TrainJob class which is responsible for training classification and regression models using Gradient Boosting Decision Trees (GBDT). The training process includes handling imbalanced datasets for classification tasks and saving the trained models.

Classes:
    TrainJob: A class that provides methods to train classification and regression models.

Methods:
    train(cls, train_data_store: DataStore, valid_data_store: DataStore, task: str = "cls") -> Booster:
        Trains a model based on the specified task ('cls' for classification or 'reg_all' for regression) and returns the trained Booster model.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import logging
from pathlib import Path

from cxsim.data.data_store import DataStore
from cxsim.trainer.gbdt import GBDT
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import Booster

logger = logging.getLogger(__name__)


SCALE_POS_WEIGHT = 5.0
MIN_DATA_IN_LEAF = 10
NUM_ITERATION = 1500
EARLY_STOPPING_ROUND = 50
VERBOSE = 2
NUM_THREADS = 4
CLS_MODEL_FILE = "cls_model.lgb"
REG_MODEL_FILE = "reg_model.lgb"


class TrainJob:
    @classmethod
    def train(
        cls, train_data_store: DataStore, valid_data_store: DataStore, task: str = "cls"
    ) -> Booster:
        """
        Trains a model based on the specified task ('cls' for classification or 'reg_all' for regression).

        Args:
            train_data_store (DataStore): The training data store.
            valid_data_store (DataStore): The validation data store.
            task (str): The task type, either 'cls' for classification or 'reg_all' for regression. Default is 'cls'.

        Returns:
            Booster: The trained Booster model.
        """
        logger.info(f"Start training for {task} task")
        save_dir = Path("data/tmp").resolve()
        if task == "cls":
            rus = RandomUnderSampler(random_state=42, sampling_strategy=0.2)
            X_train, y_train = rus.fit_resample(
                train_data_store.cls_train_data.drop(columns=["label"], axis=1),
                train_data_store.cls_train_data["label"],
            )
            X_valid, y_valid = rus.fit_resample(
                valid_data_store.cls_train_data.drop(columns=["label"], axis=1),
                valid_data_store.cls_train_data["label"],
            )
            trainer = GBDT()
            trainer.set_data(X_train, y_train, X_valid, y_valid)

            # run train
            train_params = {
                "objective": "binary",
                "scale_pos_weight": SCALE_POS_WEIGHT,
                "min_data_in_leaf": MIN_DATA_IN_LEAF,
                "num_iteration": NUM_ITERATION,
                "early_stopping_round": EARLY_STOPPING_ROUND,
                "metric": "auc",
                "verbose": VERBOSE,
                "num_threads": NUM_THREADS,
            }
            bst = trainer.train(train_params=train_params)
            trainer.save(f"{save_dir}/{CLS_MODEL_FILE}")
            logger.info(f"Model saved to {save_dir}/{CLS_MODEL_FILE}")
            logger.info(f"Finish training for {task} task")
            return bst

        elif task == "reg_all":
            X_reg_train = train_data_store.reg_train_data.drop(
                columns=["label"], axis=1
            )
            y_reg_train = train_data_store.reg_train_data["label"]
            X_reg_valid = valid_data_store.reg_train_data.drop(
                columns=["label"], axis=1
            )
            y_reg_valid = valid_data_store.reg_train_data["label"]

            reg_trainer = GBDT()
            reg_trainer.set_data(X_reg_train, y_reg_train, X_reg_valid, y_reg_valid)
            train_params = {
                "objective": "regression",
                "min_data_in_leaf": MIN_DATA_IN_LEAF,
                "num_iteration": NUM_ITERATION,
                "early_stopping_round": EARLY_STOPPING_ROUND,
                "metric": "rmse",
                "verbose": VERBOSE,
                "num_threads": NUM_THREADS,
            }
            bst = reg_trainer.train(train_params=train_params)
            reg_trainer.save(f"{save_dir}/{REG_MODEL_FILE}")
            logger.info(f"Model saved to {save_dir}/{REG_MODEL_FILE}")
            logger.info(f"Finish training for {task} task")
            return bst

        else:
            raise ValueError(f"task: {task} is not supported")
