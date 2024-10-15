"""
This script is the entry point for the CXSimulator application. It sets up command-line argument parsing,
configures logging, and executes different stages of the simulation process based on the provided arguments.

Author: Akira Kasuga
Affiliation: CyberAgent, inc.
"""

from __future__ import annotations

import argparse
import logging

from rich.logging import RichHandler

from cxsim.task import CampaignSimulation, PredictionModel

logger = logging.getLogger(__name__)


def global_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "rich"],
        help="Logging level.",
    )
    parser.add_argument(
        "--stage",
        default="preprocess",
        choices=["model", "simulation"],
        help="Stage of the process.",
    )
    parser.add_argument(
        "--use-cache", action="store_true", help="Use cache from BigQuery"
    )
    parser.add_argument(
        "--campaign-title",
        default="Enjoy 1 month Free of YouTube Premium for Youtube related Product!",
        help="Simulate your campaign title.",
    )
    parser.add_argument(
        "--use-embed-cache", action="store_true", help="Use cache from AzureOpenAI"
    )


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CXSimulator")
    global_options(parser)
    options, extra_options = parser.parse_known_args(args)
    if options.log_level == "rich":
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=getattr(logging, options.log_level.upper()), format="%(message)s"
        )

    if options.stage == "model":
        PredictionModel.train(options.use_cache)
        return

    if options.stage == "simulation":
        CampaignSimulation.run(
            promotion_name=options.campaign_title,
            use_cache=options.use_cache,
            use_embed_cache=options.use_embed_cache,
        )
        return

    else:
        raise ValueError(f"Invalid stage: {options.stage}")


# Allow the script to be run standalone (useful during development).
if __name__ == "__main__":
    main()
