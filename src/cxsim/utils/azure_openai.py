"""
This module provides a client for interacting with the Azure OpenAI service.
It includes functionality to obtain text embeddings using specified models and dimensions.

Classes:
    OpenAIClient: A client for Azure OpenAI that supports embedding generation.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import logging
from typing import Any, Dict

from cxsim.config.env import (
    AZURE_OPENAI_US_ENDPOINT,
    AZURE_OPENAI_US_KEY,
    AZURE_OPENAI_US_VERSION,
)
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self, region: str = "eastus"):
        if AZURE_OPENAI_US_ENDPOINT is None:
            raise ValueError("Azure OpenAI endpoint is not set")
        self.region = region
        if region == "eastus":
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_US_KEY,
                api_version=AZURE_OPENAI_US_VERSION,
                azure_endpoint=AZURE_OPENAI_US_ENDPOINT,
            )
        else:
            raise NotImplementedError(f"Region {region} is not implemented")

    def get_embedding(
        self,
        text: str = "The user leave a site.",
        model: str = "text-embedding-3-small",
        dimensions: int = 128,
    ) -> Dict[str, Any]:
        if self.region == "eastus":
            logger.info(text)
            response = self.client.embeddings.create(
                input=text, model=model, dimensions=dimensions
            )
        elif self.region == "japaneast":
            response = self.client.embeddings.create(input=text, model=model)
        else:
            raise ValueError(f"Invalid region: {self.region}")

        return response.model_dump()
