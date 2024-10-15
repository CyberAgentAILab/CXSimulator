"""
This module provides the EmbeddingGraph class, which extends the BaseGraphData class to generate, save, and load node embeddings using various methods, including OpenAI's embedding service.

Classes:
    EmbeddingGraph: A class to handle node embeddings for graph data.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from cxsim.data.base_graph import BaseGraphData
from cxsim.utils.azure_openai import OpenAIClient

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingGraph(BaseGraphData):
    # train/valid/test label
    split: str = "train"

    # embedding
    dimensions: int = 16
    node2embedding: dict = field(default_factory=dict)

    def __init__(self, split: str = "train"):
        super().__init__()
        self.split = split

    def get_node2embedding(self, method: str = "openai") -> None:
        """
        Generates embeddings for nodes in the graph using the specified method.
        Args:
            method (str): The method to use for generating embeddings. Default is "openai".
        Raises:
            NotImplementedError: If the specified method is not implemented.
            Exception: If an error occurs during embedding generation, a default embedding is assigned and the error is logged.
        """

        # initialize
        node2embedding = {}

        # iterate for embedding
        for node in self.graph.nodes:
            try:
                if method == "openai":
                    self.openai = OpenAIClient()
                    node2embedding[node] = self.openai.get_embedding(
                        text=node, dimensions=self.dimensions
                    )
                else:
                    raise NotImplementedError

            except Exception as e:
                node2embedding[node] = {
                    "data": [
                        {
                            "embedding": [0.0 for i in range(self.dimensions)],
                            "index": 0,
                            "object": "embedding",
                        }
                    ],
                    "model": "Embedding Error",
                    "object": "list",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
                logger.info(e)

        self.node2embedding = node2embedding

    def save_node_embedding(
        self,
        embedding_file_name: str = "train_node2embedding.json",
    ) -> None:
        """
        Save the node embeddings to a JSON file.
        Args:
            embedding_file_name (str): The name of the file to save the embeddings to. Defaults to "train_node2embedding.json".
        Returns:
            None
        """

        save_dir = Path("data/tmp").resolve()

        file_path = os.path.join(save_dir, embedding_file_name)
        with open(file_path, "w") as f:
            json.dump(self.node2embedding, f, indent=2)
        logger.info(f"save to {file_path}")

    def load_node_embedding(
        self, embedding_file_name: str = "train_node2embedding.json"
    ) -> None:
        """
        Loads node embeddings from a specified JSON file.
        Args:
            embedding_file_name (str): The name of the JSON file containing the node embeddings. Defaults to "train_node2embedding.json".
        Returns:
            None
        """

        save_dir = Path("data/tmp").resolve()
        file_path = os.path.join(save_dir, embedding_file_name)
        with open(file_path) as f:
            self.node2embedding = json.load(f)
        logger.info(f"load from {embedding_file_name}")

    def add_node2embeddings(self, src_node_name: str, use_embed_cache=False) -> None:
        """
        Adds embeddings for a specific node to the node2embedding dictionary.
        Args:
            src_node_name (str): The name of the source node to add embeddings for.
            use_embed_cache (bool): Whether to use cached embeddings. Defaults to False.
        Returns:
            None
        """
        if use_embed_cache:
            save_dir = Path("data/tmp").resolve()
            file_path = os.path.join(save_dir, "sample_campaign_embeddings.json")
            with open(file_path) as f:
                campaign_embeddings = json.load(f)
            logger.info(f"load from {file_path}")
            self.node2embedding[src_node_name] = campaign_embeddings[src_node_name]
            return

        self.openai = OpenAIClient()
        self.node2embedding[src_node_name] = self.openai.get_embedding(
            text=src_node_name, dimensions=self.dimensions
        )
