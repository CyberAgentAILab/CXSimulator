"""
This module contains the implementation of the PredictionModel and CampaignSimulation classes.
The PredictionModel class provides methods for preprocessing data and training a binary classification / regression model,
while the CampaignSimulation class provides methods for running a campaign simulation using pre-trained models.

Classes:
    PredictionModel: Handles data preprocess and the training of binary classification / regression models.
    CampaignSimulation: Handles the execution of campaign simulations using pre-trained models.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import logging

import numpy as np

from cxsim.eval.evaluation import Evaluation
from cxsim.jobs.preprocess import PreprocessJob
from cxsim.jobs.simulation import PROB_THRESHOLD, SimulationJob
from cxsim.jobs.train import TrainJob

logger = logging.getLogger(__name__)


class PredictionModel:
    @classmethod
    def train(cls, use_cache: bool = False) -> None:
        if use_cache:
            prep_job = PreprocessJob()
            prep_job.load_data(use_cache=True)
            prep_job.get_embeddings(use_cache=True)
        else:
            prep_job = PreprocessJob()
            prep_job.load_data()
            prep_job.get_embeddings()
            prep_job.save_graphs()

        # Get test data
        X_test, y_test = prep_job.get_test_data()

        # Binary Classification for the first layer
        prep_job.get_train_data(task="cls")
        cls_bst = TrainJob.train(
            train_data_store=prep_job.train_data_store,
            valid_data_store=prep_job.valid_data_store,
            task="cls",
        )

        y_pred = np.clip(cls_bst.predict(X_test), 0.0, 1.0)
        therehold = 0.1  # move to config file
        y_true_cls = np.array([1.0 if label >= therehold else 0.0 for label in y_test])
        y_pred_cls = np.array([1.0 if label >= therehold else 0.0 for label in y_pred])
        logger.info(Evaluation.evaluate_binary(y_true_cls, y_pred_cls, y_pred))

        # Regression for the second layer
        prep_job.get_train_data(task="reg_all")
        reg_bst = TrainJob.train(
            train_data_store=prep_job.train_data_store,
            valid_data_store=prep_job.valid_data_store,
            task="reg_all",
        )

        pred_cls_probs = cls_bst.predict(X_test)
        pred_reg_X = prep_job.train_data_store.convert_cls_to_reg(
            pred_cls_probs, X_test, PROB_THRESHOLD
        )
        pred_reg_probs = np.clip(reg_bst.predict(pred_reg_X), 0.0, 1.0)
        y_pred = prep_job.train_data_store.merge_probs(
            pred_cls_probs, pred_reg_probs, PROB_THRESHOLD
        )
        logger.info(Evaluation.evaluate_regression(y_test, y_pred))


class CampaignSimulation:
    @classmethod
    def run(
        cls,
        promotion_name: str,
        use_cache: bool = True,
        use_embed_cache: bool = False,
    ) -> None:
        sim_job = SimulationJob()
        sim_job.load_data(use_cache=use_cache)
        sim_job.load_model("cls_model.lgb", "reg_model.lgb")
        result = sim_job.get_simulation_result(
            promotion_name=promotion_name, use_embed_cache=use_embed_cache
        )

        logger.info(result)
