"""
This module defines the DataStore class, which extends the EmbeddingGraph class to manage and process training data for classification and regression tasks.
It includes methods for generating training data, performing inference, converting classification probabilities to regression data, merging probabilities, and generating prediction graphs.
Additionally, it provides functionality for simulating path conversions using multiprocessing.

Classes:
    DataStore: A class for managing and processing training data for classification and regression tasks.

Functions:
    wrap_func(args): A wrapper function for multiprocessing.
    generate_index(array, seed, max_length, index): Generates a sequence of indices based on transition probabilities.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import itertools
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, List

import networkx as nx
import numpy as np
import pandas as pd
from cxsim.data.embedding_graph import EmbeddingGraph

logger = logging.getLogger(__name__)


@dataclass
class DataStore(EmbeddingGraph):
    cls_train_data: pd.DataFrame = pd.DataFrame()
    reg_train_data: pd.DataFrame = pd.DataFrame()
    pred_graph: nx.DiGraph = nx.DiGraph()

    def __init__(self, split: str = "train"):
        super().__init__(split=split)

    def get_train_data(self, task: str = "cls") -> None:
        """
        Generates training data for different tasks and stores it in the instance variables.
        Parameters:
        task (str): The type of task for which to generate training data.
                    It can be one of the following:
                    - "cls": For classification tasks. Generates a DataFrame `cls_train_data`
                             with node embeddings and labels indicating the presence of edges.
                    - "reg": For regression tasks. Generates a DataFrame `reg_train_data`
                             with node embeddings and edge weights as labels.
                    - "reg_all": For regression tasks considering all possible node pairs.
                                 Generates a DataFrame `reg_train_data` with node embeddings
                                 and edge weights as labels, with non-existent edges labeled as 0.0.
        Raises:
        NotImplementedError: If the provided task is not supported.
        """

        data = []
        num_graph_features = 0

        if task == "cls":
            nodes = list(self.graph.nodes)
            for u, v in itertools.product(nodes, nodes):
                if u != self.state_regularize and v != self.state_regularize:
                    if self.graph.has_edge(u, v):
                        label = 1.0
                    else:
                        label = 0.0
                    row = [
                        *self.node2embedding[u]["data"][0]["embedding"],
                        *self.node2embedding[v]["data"][0]["embedding"],
                        label,
                    ]
                    data.append(row)
            dimensions = len(
                self.node2embedding[self.state_init]["data"][0]["embedding"]
            )

            self.cls_train_data = pd.DataFrame(
                data,
                columns=[
                    f"embedding_{i}"
                    for i in range((dimensions + num_graph_features) * 2)
                ]
                + ["label"],
            )

        elif task == "reg":
            for edge in self.graph.edges(data=True):
                # edge[0]:src edge[1]:dst edge[2]: attributes
                label = edge[2][self.weight_name]
                # remove regularize state node
                if edge[1] != self.state_regularize:
                    row = [
                        *self.node2embedding[edge[0]]["data"][0]["embedding"],
                        *self.node2embedding[edge[1]]["data"][0]["embedding"],
                        label,
                    ]
                    data.append(row)
            dimensions = len(
                self.node2embedding[self.state_init]["data"][0]["embedding"]
            )

            self.reg_train_data = pd.DataFrame(
                data,
                columns=[
                    f"embedding_{i}"
                    for i in range((dimensions + num_graph_features) * 2)
                ]
                + ["label"],
            )

        elif task == "reg_all":
            nodes = list(self.graph.nodes)
            for u, v in itertools.product(nodes, nodes):
                if u != self.state_regularize and v != self.state_regularize:
                    if self.graph.has_edge(u, v):
                        label = self.graph.edges[u, v][self.weight_name]
                    else:
                        label = 0.0
                    row = [
                        *self.node2embedding[u]["data"][0]["embedding"],
                        *self.node2embedding[v]["data"][0]["embedding"],
                        label,
                    ]
                    data.append(row)
            dimensions = len(
                self.node2embedding[self.state_init]["data"][0]["embedding"]
            )

            self.reg_train_data = pd.DataFrame(
                data,
                columns=[
                    f"embedding_{i}"
                    for i in range((dimensions + num_graph_features) * 2)
                ]
                + ["label"],
            )

        else:
            raise NotImplementedError(f"task is not supported: {task}")

    def get_inference_data(
        self, src_node_name: str, direction: str = "predecessor", nodes: List[str] = []
    ) -> pd.DataFrame:
        """
        Retrieves inference data for a given source node and a list of target nodes.
        Parameters:
        src_node_name (str): The name of the source node.
        direction (str, optional): The direction of the relationship between nodes.
                       It can be either "predecessor" or "successor".
                       Defaults to "predecessor".
        nodes (List[str], optional): A list of target node names. If not provided,
                         all nodes in the graph are considered.
                         Defaults to an empty list.
        Returns:
        pd.DataFrame: A DataFrame containing the concatenated embeddings of the source
                  and target nodes. The columns are named as "embedding_i" where
                  i ranges from 0 to (2 * dimensions - 1).
        Raises:
        NotImplementedError: If the direction is not "predecessor" or "successor".
        """

        data = []

        if nodes == []:
            nodes = list(self.graph.nodes)
        for dst_node_name in nodes:
            if dst_node_name != self.state_regularize:
                if direction == "predecessor":
                    row = [
                        *self.node2embedding[src_node_name]["data"][0]["embedding"],
                        *self.node2embedding[dst_node_name]["data"][0]["embedding"],
                    ]
                elif direction == "successor":
                    row = [
                        *self.node2embedding[dst_node_name]["data"][0]["embedding"],
                        *self.node2embedding[src_node_name]["data"][0]["embedding"],
                    ]
                else:
                    raise NotImplementedError(
                        f"direction is not supported: {direction}"
                    )

                data.append(row)
        df = pd.DataFrame(
            data, columns=[f"embedding_{i}" for i in range(self.dimensions * 2)]
        )

        return df

    def convert_cls_to_reg(
        self,
        cls_pred_probs: np.ndarray[Any, Any] | Any | List[Any],
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Converts classification predictions to regression format based on a threshold.
        Parameters:
        -----------
        cls_pred_probs : np.ndarray[Any, Any] | Any | List[Any]
            The classification prediction probabilities.
        X : pd.DataFrame
            The input DataFrame containing features.
        threshold : float, optional
            The threshold value to filter classification predictions (default is 0.5).
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing rows from X where classification predictions exceed the threshold.
        Raises:
        -------
        NotImplementedError
            If the type of cls_pred_probs is not supported.
        """
        if isinstance(cls_pred_probs, np.ndarray):
            _X = X[cls_pred_probs > threshold]
            return _X
        else:
            raise NotImplementedError(
                f"cls_pred_probs type is not supported: {type(cls_pred_probs)}"
            )

    def merge_probs(
        self,
        cls_pred_probs: np.ndarray[Any, Any] | Any | List[Any],
        reg_pred_probs: np.ndarray[Any, Any] | Any | List[Any],
        threshold: float = 0.5,
    ) -> np.ndarray[Any, Any]:
        """
        Merges classification and regression prediction probabilities based on a threshold.
        Args:
            cls_pred_probs (np.ndarray[Any, Any] | Any | List[Any]): Classification prediction probabilities.
            reg_pred_probs (np.ndarray[Any, Any] | Any | List[Any]): Regression prediction probabilities.
            threshold (float, optional): Threshold for classification probabilities. Defaults to 0.5.
        Returns:
            np.ndarray[Any, Any]: Merged prediction probabilities.
        Raises:
            NotImplementedError: If the input types for cls_pred_probs or reg_pred_probs are not supported.
        """
        if isinstance(cls_pred_probs, np.ndarray) and isinstance(
            reg_pred_probs, np.ndarray
        ):
            pred_probs = np.array([0.0 for i in range(len(cls_pred_probs))])
            np.put(pred_probs, np.where(cls_pred_probs > threshold)[0], reg_pred_probs)
            return pred_probs
        else:
            raise NotImplementedError(
                f"cls_pred_probs type is not supported: {type(cls_pred_probs)} \
                    or reg_pred_probs type is not supported: {type(reg_pred_probs)}"
            )

    def get_pred_graph(
        self,
        src_node_name: str,
        pred_probs: np.ndarray,
        succ_probs: np.ndarray,
        gamma: float = 1.0,
    ) -> None:
        """
        Generates a prediction graph by updating edge weights based on given probabilities.
        Args:
            src_node_name (str): The source node name.
            pred_probs (np.ndarray): Array of predecessor probabilities.
            succ_probs (np.ndarray): Array of successor probabilities.
            gamma (float, optional): A scaling factor for successor probabilities. Defaults to 1.0.
        Returns:
            None
        """
        edges = []

        # from new
        norm_pred_probs = pred_probs / pred_probs.sum()
        for dst_node_name, norm_pred_prob in zip(self.graph.nodes, norm_pred_probs):
            from_edge = [src_node_name, dst_node_name, norm_pred_prob]
            edges.append(from_edge)

        # to new
        for dst_node_name, succ_prob in zip(self.graph.nodes, succ_probs):
            out_edges = self.graph.out_edges(dst_node_name, data=True)
            deno = 1.0 + gamma * succ_prob
            to_edges = [
                [out_edge[0], out_edge[1], (out_edge[2][self.weight_name] / deno)]
                for out_edge in out_edges
            ]
            edges.extend(to_edges)
            edges.append([dst_node_name, src_node_name, gamma * succ_prob / deno])

        self.pred_graph = self.graph.copy()
        # if exits, update edge weight. if not, add edge.
        self.pred_graph.add_weighted_edges_from(edges, weight=self.weight_name)

    def generate_path_conversion_df(
        self,
        num: int = 10,
        max_length: int = 15,
        pred: bool = False,
        fluc: int = 0,
    ) -> pd.DataFrame:
        """
        Generates a DataFrame containing simulated path conversions.

        Parameters:
        -----------
        num : int, optional
            Number of samples to simulate (default is 10).
        max_length : int, optional
            Maximum length of each simulated path (default is 15).
        pred : bool, optional
            If True, use the predicted graph for simulation; otherwise, use the actual graph (default is False).
        fluc : int, optional
            Fluctuation parameter to adjust the simulation (default is 0).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the simulated paths with each path represented as a string.

        Notes:
        ------
        - The function logs the start and end of the simulation process.
        - Utilizes multiprocessing to parallelize the simulation.
        """
        logger.info(f"Simulating... {num} samples")

        if pred:
            tm = nx.to_numpy_array(
                self.pred_graph, dtype=np.float64, weight=self.weight_name
            )
            states = list(self.pred_graph.nodes)
            init_index = states.index(self.state_init)
        else:
            tm = nx.to_numpy_array(
                self.graph, dtype=np.float64, weight=self.weight_name
            )
            states = list(self.graph.nodes)
            init_index = states.index(self.state_init)

        pool = mp.Pool(mp.cpu_count())
        result = pool.map(
            wrap_func, [(tm, i + fluc, max_length, init_index) for i in range(num)]
        )

        result = [[states[index] for index in session] for session in result]
        rows = [{"path": " > ".join(path)} for path in result]

        df = pd.DataFrame(rows)
        logger.info("Simulattion was done.")

        return df


# for multiprocessing
def wrap_func(args):
    return generate_index(*args)


def generate_index(
    array: np.ndarray, seed: int, max_length: int = 15, index: int = 0
) -> List[int]:
    """
    Generates a sequence of indices based on the given probability array.
    Args:
        array (np.ndarray): A 2D array where each row represents the probability distribution for the next index.
        seed (int): Seed for the random number generator to ensure reproducibility.
        max_length (int, optional): Maximum length of the generated sequence. Defaults to 15.
        index (int, optional): Starting index for the sequence. Defaults to 0.
    Returns:
        List[int]: A list of indices representing the generated sequence.
    """

    index_list = [i for i in range(array.shape[0])]
    sequences = [index]
    rst = np.random.RandomState(seed)

    for iter in range(max_length):
        prob = array[index]
        try:
            next_index = rst.choice(index_list, 1, p=prob)
            sequences.append(next_index[0])
            index = next_index[0]
        except Exception:
            break

    return sequences
