"""
This module provides a utility class for interacting with Google BigQuery.
It allows querying BigQuery and retrieving results as a pandas DataFrame.

Classes:
    Bigquery: A class to handle BigQuery operations.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

import pandas as pd
from cxsim.config.env import GOOGLE_CLOUD_PROJECT_ID
from google.cloud import bigquery


class Bigquery:
    def __init__(
        self, project: str | None = GOOGLE_CLOUD_PROJECT_ID
    ):  # TODO: replace with your project id
        self.client = bigquery.Client(project=project)

    def get_dataframe(self, sql: str) -> pd.DataFrame:
        df = self.client.query_and_wait(sql).to_dataframe()
        return df
