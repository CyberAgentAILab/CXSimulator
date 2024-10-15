"""
This module provides the BaseGraphData class for handling graph data operations.
It includes methods to load and save graphs from JSON files, BigQuery, and CSV files,
and to regularize the graph by adjusting edge weights.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List

import networkx as nx
from cxsim.utils.gcp import Bigquery

logger = logging.getLogger(__name__)


@dataclass
class BaseGraphData:
    # base graph
    data_path: str = "train_graph.json"
    graph: nx.DiGraph = nx.DiGraph()
    weight_name: str = "probability"

    # define state
    state_regularize: str = "undefined"
    state_init: str = "session_start"
    state_goal: str = "purchase"
    state_end: List[str] = field(default_factory=list)

    def __init__(self):
        self.openai = None
        self.bq_client = None

    def load_graph(self, data_path: str) -> None:
        """
        Load a graph from a JSON file.
        This method reads a JSON file from the specified path, parses it, and
        converts it into a NetworkX graph using the node-link format.
        Args:
            data_path (str): The path to the JSON file containing the graph data.
        Returns:
            None
        """

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.graph = nx.node_link_graph(data)

    def save_graph(self, data_path: str) -> None:
        """
        Save the current graph to a file in JSON format.
        Args:
            data_path (str): The path to the file where the graph will be saved.
        Returns:
            None
        """

        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(self.graph), f, indent=2)

    def load_graph_from_bigquery(
        self,
        sql_path: str,
        start_date: str = "20170701",
        end_date: str = "20170701",
        alpha: int = 10,
    ) -> None:
        """
        Loads a directed graph from a BigQuery SQL query.
        Args:
            sql_path (str): Path to the SQL file containing the query.
            start_date (str, optional): Start date for the query in 'YYYYMMDD' format. Defaults to "20170701".
            end_date (str, optional): End date for the query in 'YYYYMMDD' format. Defaults to "20170701".
            alpha (int, optional): An integer parameter to be used in the query. Defaults to 10.
        Returns:
            None
        """
        with open(sql_path) as f:
            sql = f.read()
        sql = sql.format(start_date=start_date, end_date=end_date, alpha=alpha)
        logger.debug(sql)

        self.bq_client = Bigquery()
        df = self.bq_client.get_dataframe(sql)
        self.graph = nx.from_pandas_edgelist(
            df, edge_attr=True, create_using=nx.DiGraph()
        )
        self.regularize_graph()

    def regularize_graph(self) -> None:
        """
        Regularizes the graph by ensuring that the sum of the weights of the outgoing edges
        from each node equals 1. If the sum is less than 1, an edge is added from the node
        to a state regularize node with the remaining weight.
        Returns:
            None
        """

        undefined_edges = []
        for src_node_name in self.graph.nodes:
            out_edges = self.graph.out_edges(src_node_name, data=True)
            out_weight_sum = sum([edge[2]["weight"] for edge in out_edges])
            undefined_edges.append(
                (src_node_name, self.state_regularize, 1 - out_weight_sum)
            )

        self.graph.add_weighted_edges_from(undefined_edges, weight=self.weight_name)
