"""
This module defines the PreprocessJob class, which is responsible for preprocessing data for training, validation, and testing in a machine learning pipeline. The class handles loading data from BigQuery or cache, generating embeddings, saving graphs, and preparing training and test datasets.

Classes:
    PreprocessJob: A class to preprocess data for machine learning tasks.

Methods:
    __init__(): Initializes the PreprocessJob with train, validation, and test data stores.
    define_state(split: str) -> DataStore: Defines the state for a given data split.
    load_data(use_cache: bool = False) -> None: Loads data from BigQuery or cache.
    get_embeddings(use_cache: bool = False) -> None: Generates or loads embeddings for the data.
    save_graphs() -> None: Saves the graphs to temporary files.
    get_train_data(task: str) -> None: Prepares training data for a specified task.
    get_test_data() -> tuple[pd.DataFrame, pd.Series]: Prepares and returns test data.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from cxsim.data.data_store import DataStore

logger = logging.getLogger(__name__)

SQL_FILE = "ga360.sql"
TRAIN_START_DATE = "20160801"
TRAIN_END_DATE = "20160831"
VALID_START_DATE = "20160901"
VALID_END_DATE = "20160930"
TEST_START_DATE = "20161001"
TEST_END_DATE = "20161031"
TRAIN_GRAPH_FILE = "train_graph.json"
TRAIN_EMBEDDING_FILE = "train_node2embedding.json"
VALID_GRAPH_FILE = "valid_graph.json"
VALID_EMBEDDING_FILE = "valid_node2embedding.json"
TEST_GRAPH_FILE = "test_graph.json"
TEST_EMBEDDING_FILE = "test_node2embedding.json"
ALPHA = 3
ZERO_SAMPLE_SIZE = 300
NON_ZERO_SAMPLE_SIZE = 700


@dataclass
class PreprocessJob:
    train_data_store: DataStore = DataStore()
    valid_data_store: DataStore = DataStore()
    test_data_store: DataStore = DataStore()
    sql_file_dir: str | os.PathLike = Path("data/sqls").resolve()
    tmp_file_dir: str | os.PathLike = Path("data/tmp").resolve()

    def __init__(self):
        self.train_data_store = self.define_state("train")
        self.valid_data_store = self.define_state("valid")
        self.test_data_store = self.define_state("test")

    @staticmethod
    def define_state(split: str) -> DataStore:
        """
        Initializes and returns a DataStore object with predefined states and attributes.
        Args:
            split (str): The split identifier for the DataStore.
        Returns:
            DataStore: An initialized DataStore object with specified attributes.
        """

        data_store = DataStore(split=split)
        data_store.weight_name = "weight"
        data_store.state_init = "session_start"
        data_store.state_end = ["session_end", "undefined"]
        data_store.state_goal = '{"actionType":"Check out","pageTitle":"Checkout Your Information","pagePath":null,"productName":null}'
        return data_store

    def load_data(self, use_cache: bool = False) -> None:
        """
        Load GA360 data into the train, validation, and test data stores.
        If `use_cache` is True, the method loads preprocessed graph data from cache files.
        Otherwise, it loads data from BigQuery Public Datasets for specified date ranges.
        Args:
            use_cache (bool): Flag to determine whether to load data from cache. Defaults to False.
        Returns:
            None
        """

        # load GA360 data
        if use_cache:
            logger.info("load a Graph from a cache ...")
            self.train_data_store.load_graph(f"{self.tmp_file_dir}/{TRAIN_GRAPH_FILE}")
            self.valid_data_store.load_graph(f"{self.tmp_file_dir}/{VALID_GRAPH_FILE}")
            self.test_data_store.load_graph(f"{self.tmp_file_dir}/{TEST_GRAPH_FILE}")
            logger.info("complete loading a Graph")
            return

        logger.info("load GA360 data from BigQuery Public Datasets...")
        self.train_data_store.load_graph_from_bigquery(
            sql_path=f"{self.sql_file_dir}/{SQL_FILE}",
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            alpha=ALPHA,
        )
        self.valid_data_store.load_graph_from_bigquery(
            sql_path=f"{self.sql_file_dir}/{SQL_FILE}",
            start_date=VALID_START_DATE,
            end_date=VALID_END_DATE,
            alpha=ALPHA,
        )
        self.test_data_store.load_graph_from_bigquery(
            sql_path=f"{self.sql_file_dir}/{SQL_FILE}",
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            alpha=ALPHA,
        )
        logger.info("complete loading GA360 data.")

    def get_embeddings(self, use_cache: bool = False) -> None:
        """
        Generate or load node embeddings for train, validation, and test datasets.

        If `use_cache` is True, the function loads precomputed embeddings from cache files.
        Otherwise, it computes the embeddings and saves them to cache files.

        Args:
            use_cache (bool): Flag to determine whether to load embeddings from cache. Defaults to False.

        Returns:
            None
        """

        if use_cache:
            logger.info("load embeddings from a cache ...")
            self.train_data_store.load_node_embedding(TRAIN_EMBEDDING_FILE)
            self.valid_data_store.load_node_embedding(VALID_EMBEDDING_FILE)
            self.test_data_store.load_node_embedding(TEST_EMBEDDING_FILE)
            logger.info("complete loading embeddings")
            return

        self.train_data_store.get_node2embedding()
        self.valid_data_store.get_node2embedding()
        self.test_data_store.get_node2embedding()

        self.train_data_store.save_node_embedding(TRAIN_EMBEDDING_FILE)
        self.valid_data_store.save_node_embedding(VALID_EMBEDDING_FILE)
        self.test_data_store.save_node_embedding(TEST_EMBEDDING_FILE)

    def save_graphs(self) -> None:
        """
        Saves the train, validation, and test graphs to temporary files.

        This method saves the graph data for the train, validation, and test datasets
        to their respective temporary files defined by the class attributes.

        Returns:
            None
        """
        self.train_data_store.save_graph(f"{self.tmp_file_dir}/{TRAIN_GRAPH_FILE}")
        self.valid_data_store.save_graph(f"{self.tmp_file_dir}/{VALID_GRAPH_FILE}")
        self.test_data_store.save_graph(f"{self.tmp_file_dir}/{TEST_GRAPH_FILE}")

    def get_train_data(self, task: str) -> None:
        """
        Retrieves training, validation, and test data based on the specified task.
        Args:
            task (str): The type of task for which to retrieve data.
                        Supported values are "cls" for classification and "reg_all" for regression.
        Raises:
            ValueError: If the specified task is not supported.
        """

        if task == "cls":
            self.train_data_store.get_train_data(task="cls")
            self.valid_data_store.get_train_data(task="cls")
            self.test_data_store.get_train_data(task="cls")
        elif task == "reg_all":
            self.train_data_store.get_train_data(task=task)
            self.valid_data_store.get_train_data(task=task)
            self.test_data_store.get_train_data(task="reg_all")
        else:
            raise ValueError(f"task: {task} is not supported")

    def get_test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Retrieves and samples test data from the test data store.
        The function samples a specified number of zero and non-zero label data points
        from the training data stored in `self.test_data_store`. It then concatenates
        these samples to form the test dataset.
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing the test features
            (X_test) and the test labels (y_test).
        """

        self.test_data_store.get_train_data("reg_all")

        zero_test_data = self.test_data_store.reg_train_data[
            self.test_data_store.reg_train_data["label"] == 0.0
        ].sample(ZERO_SAMPLE_SIZE)
        non_zero_test_data = self.test_data_store.reg_train_data[
            self.test_data_store.reg_train_data["label"] != 0.0
        ].sample(NON_ZERO_SAMPLE_SIZE)
        test_data = pd.concat([zero_test_data, non_zero_test_data])
        X_test = test_data.drop(columns=["label"], axis=1)
        y_test = test_data["label"]

        return X_test, y_test
