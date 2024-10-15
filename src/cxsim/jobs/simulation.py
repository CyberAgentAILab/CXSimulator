"""
This module defines the SimulationJob class, which is responsible for running simulations
based on graph data and node embeddings. It includes methods for loading data, loading models,
and running simulations to evaluate the impact of promotions on conversion rates.

Classes:
    SimulationJob: Handles the simulation process including data loading, model loading,
                   and running simulations.

Methods:
    __init__(): Initializes the SimulationJob with default parameters.
    load_data(use_cache: bool = True): Loads graph data and node embeddings, either from cache or BigQuery.
    load_model(cls_model_file: str, reg_model_file: str): Loads classification and regression models.
    run_simulation(promotion_name: str, num_sessions: int = 100000, cv_name: str = "Checkout Your Information", use_embed_cache: bool = False) -> float:
                   Runs the simulation for a given promotion and calculates the lift and difference in conversion rates.
    get_simulation_result(promotion_name: str, num_sessions: int = 10000, cv_name: str = "Checkout Your Information", use_embed_cache: bool = False) -> Dict[str, Any]:
                   Returns the simulation result as a dictionary.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from cxsim.data.data_store import DataStore
from cxsim.jobs.preprocess import (
    ALPHA,
    SQL_FILE,
    TRAIN_EMBEDDING_FILE,
    TRAIN_GRAPH_FILE,
    PreprocessJob,
)
from cxsim.trainer.gbdt import GBDT

logger = logging.getLogger(__name__)

PROB_THRESHOLD = 0.1
SIMULATION_START_DATE = "20161101"
SIMULATION_END_DATE = "20161130"
GAMMA = 1.0


@dataclass
class SimulationJob:
    data_store: DataStore = DataStore()
    tmp_file_dir: str | os.PathLike = Path("data/tmp").resolve()
    graph_file: str = TRAIN_GRAPH_FILE
    node2embedding_file: str = TRAIN_EMBEDDING_FILE
    cls_trainer: GBDT = GBDT()
    reg_trainer: GBDT = GBDT()

    def __init__(self):
        self.data_store = PreprocessJob.define_state("simulation")

    def load_data(self, use_cache: bool = True) -> None:
        """
        Load data for the simulation.

        Args:
            use_cache (bool): If True, load data from cache. If False, load data from BigQuery Public Datasets.

        Returns:
            None
        """

        if use_cache:
            logger.info("load data from a cache ...")
            self.data_store.load_graph(f"{self.tmp_file_dir}/{self.graph_file}")
            self.data_store.load_node_embedding(self.node2embedding_file)
            logger.info("complete loading data")
            return

        logger.info("load GA360 data from BigQuery Public Datasets...")
        self.data_store.load_graph_from_bigquery(
            sql_path=f"data/sqls/{SQL_FILE}",
            start_date=SIMULATION_START_DATE,
            end_date=SIMULATION_END_DATE,
            alpha=ALPHA,
        )
        logger.info("get embeddings ...")
        self.data_store.get_node2embedding()
        logger.info("complete.")

    def load_model(self, cls_model_file: str, reg_model_file: str) -> None:
        """
        Load classification and regression models.

        Args:
            cls_model_file (str): Path to the classification model file.
            reg_model_file (str): Path to the regression model file.

        Returns:
            None
        """
        self.cls_trainer = GBDT()
        self.cls_trainer.load(f"{self.tmp_file_dir}/{cls_model_file}")

        self.reg_trainer = GBDT()
        self.reg_trainer.load(f"{self.tmp_file_dir}/{reg_model_file}")

    def run_simulation(
        self,
        promotion_name: str,
        num_sessions: int = 100000,
        cv_name: str = "Checkout Your Information",
        use_embed_cache: bool = False,
    ) -> float:
        """
        Runs a simulation to evaluate the impact of a promotion on conversion rates.

        Args:
            promotion_name (str): The name of the promotion to simulate.
            num_sessions (int, optional): The number of sessions to simulate. Defaults to 10000.
            cv_name (str, optional): The name of the conversion event to track. Defaults to "Checkout Your Information".

        Returns:
            float: The percentage difference in conversion rates between the treatment and control groups.
        """
        # define new node
        src_node_name = f'{{"actionType":"Click a campaign page","pageTitle":null,"pagePath":null,"productName":["{promotion_name}"]}}'

        # predecessor
        self.data_store.add_node2embeddings(src_node_name, use_embed_cache)
        pred_cls_X = self.data_store.get_inference_data(
            src_node_name, direction="predecessor"
        )
        pred_cls_probs = self.cls_trainer.bst.predict(pred_cls_X)
        pred_reg_X = self.data_store.convert_cls_to_reg(
            pred_cls_probs, pred_cls_X, PROB_THRESHOLD
        )
        pred_reg_probs = np.clip(self.reg_trainer.bst.predict(pred_reg_X), 0.0, 1.0)
        pred_probs = self.data_store.merge_probs(
            pred_cls_probs, pred_reg_probs, PROB_THRESHOLD
        )

        # predecessor
        succ_cls_X = self.data_store.get_inference_data(
            src_node_name, direction="successor"
        )
        succ_cls_probs = self.cls_trainer.bst.predict(succ_cls_X)
        succ_reg_X = self.data_store.convert_cls_to_reg(
            succ_cls_probs, succ_cls_X, PROB_THRESHOLD
        )
        succ_reg_probs = np.clip(self.reg_trainer.bst.predict(succ_reg_X), 0.0, 1.0)
        succ_probs = self.data_store.merge_probs(
            succ_cls_probs, succ_reg_probs, PROB_THRESHOLD
        )

        # generate pred graph
        self.data_store.get_pred_graph(
            src_node_name, pred_probs, succ_probs, gamma=GAMMA
        )
        control_df = self.data_store.generate_path_conversion_df(
            num=num_sessions, pred=False
        )
        treatment_df = self.data_store.generate_path_conversion_df(
            num=num_sessions, pred=True
        )

        c_cv = len(control_df[control_df["path"].str.contains(cv_name, regex=False)])
        t_cv = len(
            treatment_df[treatment_df["path"].str.contains(cv_name, regex=False)]
        )
        lift = round((t_cv / c_cv), 6)
        diff = round(((lift - 1) * 100), 6)
        logger.info(
            f"""
            Promotion Name: {promotion_name}
            Control CV: {c_cv}
            Treatment CV: {t_cv}
            LIFT: {lift}
            DIFF: {diff}
            """
        )
        return diff

    def get_simulation_result(
        self,
        promotion_name: str,
        num_sessions: int = 10000,
        cv_name: str = "Checkout Your Information",
        use_embed_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Get the simulation result for a given promotion.

        Args:
            promotion_name (str): The name of the promotion to simulate.
            num_sessions (int, optional): The number of sessions to simulate. Defaults to 10000.
            cv_name (str, optional): The name of the conversion event to track. Defaults to "Checkout Your Information".
            use_embed_cache (bool, optional): Whether to use cached embeddings. Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary containing the promotion name and conversion rate difference.
        """
        diff = self.run_simulation(
            promotion_name, num_sessions, cv_name, use_embed_cache
        )
        result = {
            "Title": promotion_name,
            "CVR(float)": diff,
        }
        return result
