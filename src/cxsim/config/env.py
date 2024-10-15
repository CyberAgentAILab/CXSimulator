"""
This module is responsible for loading environment variables from a .env file
and setting up configuration constants for Azure OpenAI and Google BigQuery services.

Attributes:
    AZURE_OPENAI_US_ENDPOINT (str): The endpoint for Azure OpenAI service in the US.
    AZURE_OPENAI_US_VERSION (str): The version of the Azure OpenAI service.
    AZURE_OPENAI_US_KEY (str): The API key for accessing the Azure OpenAI service.
    GOOGLE_CLOUD_PROJECT_ID (str): The project ID for Google Cloud's BigQuery service.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import os

from dotenv import load_dotenv

env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(verbose=True, dotenv_path=env_file_path)

# Azure OpenAI
AZURE_OPENAI_US_ENDPOINT = os.getenv("AZURE_OPENAI_US_ENDPOINT")
AZURE_OPENAI_US_VERSION = os.getenv("AZURE_OPENAI_US_VERSION")
AZURE_OPENAI_US_KEY = os.getenv("AZURE_OPENAI_US_KEY")

# Google BigQuery
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
