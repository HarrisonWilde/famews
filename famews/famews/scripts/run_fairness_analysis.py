import argparse
import logging
import socket
from pathlib import Path

import coloredlogs
import torch

from famews.fairness_check.analysis_pipeline import FairnessAnalysisPipeline
from famews.train.pipeline import SetupGin

LOGGING_LEVEL = logging.INFO


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # ------------------
    # Preprocessing Options
    # ------------------
    parser.add_argument(
        "--seed",
        dest="seed",
        default=42,
        required=False,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "-g", "--gin_config_path", required=True, type=Path, help="Path to Gin configuration file"
    )
    parser.add_argument(
        "-l", "--log_dir", required=True, type=Path, help="Path to log directory for this run"
    )
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    """Fairness Analysis Script procedure"""

    # get GPU availability
    cuda_available = torch.cuda.is_available()
    device_string = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    logging.info(40 * "=")
    logging.info("Start Training script")
    logging.info(f"Host: {socket.gethostname()}")
    logging.info(f"Torch device: {device_string}")
    if cuda_available:
        gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        logging.info(f"GPU Memory: {gpu_memory} GB")
    logging.info(40 * "=")

    # Parse CMD arguments
    args = parse_arguments(argv)

    # ------------------------
    # Pipeline
    # ------------------------
    arguments = vars(args)
    log_dir = arguments["log_dir"]

    setup_gin_stage = SetupGin(gin_config_files=[arguments["gin_config_path"]])
    setup_gin_stage.run()

    fairness_analysis_pipeline = FairnessAnalysisPipeline(log_dir=log_dir)
    fairness_analysis_pipeline.run()

    # ------------------------
    # Cleanup
    # ------------------------
    logging.info(40 * "=")
    logging.info("Finished")
    logging.info(40 * "=")


# ========================
# SCRIPT ENTRY
# ========================
if __name__ == "__main__":
    # set logging
    logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s - %(levelname)s | %(message)s")
    coloredlogs.install(level=LOGGING_LEVEL)

    # run script
    main()
