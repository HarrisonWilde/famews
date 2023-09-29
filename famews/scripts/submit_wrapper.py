#!/usr/bin/env python3

# ===========================================================
#
# Run a hyperparameter search
#
# ===========================================================
import argparse
import datetime
import logging
import os
import random
import shlex
import shutil
import subprocess
import time
from itertools import product
from pathlib import Path
from typing import Optional

import coloredlogs
import yaml
from coolname import generate_slug
from tqdm import tqdm

# ========================
# GLOBAL
# ========================
LOGGING_LEVEL = "INFO"
EXPERIMENT_GLOBALS = {"base_gin", "run_command", "task", "seeds", "wandb_project"}

# ========================
# Argparse
# ========================
def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    # ------------------
    # General options
    # ------------------
    parser.add_argument("-n", "--name", type=str, default="test", help="Experiment Name")
    parser.add_argument("-d", "--directory", type=Path, default="./logs")
    parser.add_argument("--clear_directory", default=False, action="store_true")
    parser.add_argument(
        "--silent",
        default=False,
        action="store_true",
        help="If `clear_directory` is not provided, this flag will prevent the user prompt",
    )
    parser.add_argument("--config", required=True, type=Path, help="Configuration yaml file")
    parser.add_argument("--print", default=False, action="store_true")
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Seconds to wait between submissions"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=-1,
        help="Number of runs to perform; default=-1 run all configs",
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="Shuffle configs before running; e.g. random search",
    )

    parser_compute = parser.add_argument_group("Compute")
    parser_compute.add_argument("-g", "--gpus", type=int, default=0)
    parser_compute.add_argument("--gpu_type", type=str, default=None)
    parser_compute.add_argument("-c", "--cores", type=int, default=2)
    parser_compute.add_argument(
        "-m", "--memory", type=int, default=4, help="Requests {m}G of memory per core"
    )
    parser_compute.add_argument(
        "-t", "--time", type=int, default=1, help="Request {t} hours of job duration"
    )
    parser_compute.add_argument(
        "-e", "--exclude", default=[], nargs="+", help="Nodes to exclude from submission"
    )

    args = parser.parse_args()
    return args


def create_slurm_command(args: argparse.Namespace, compute_config: dict = {}) -> tuple[str, bool]:
    """
    Create a slurm submit command from args and a given configuration
    of compute resources

    Parameter
    ---------
    args: argparse.Namespace
        command line arguments
    compute_config: dict
        compute resource configuration, passed by yaml config
    """

    args_dict = vars(args)
    for key in ["gpus", "gpu_type", "cores", "memory", "time"]:
        if key not in compute_config:
            compute_config[key] = args_dict[key]

    cmd = "sbatch"
    cmd += f" --time {compute_config['time']}:00:00"
    cmd += f" -c {compute_config['cores']}"

    memory = int(compute_config["memory"]) * int(compute_config["cores"])
    cmd += f" --mem {memory}000"

    if len(args.exclude) > 0:
        nodelist = " ".join(args.exclude)
        logging.info(f"[COMPUTE] excluding nodes: {nodelist}")
        cmd += f" --exclude {nodelist}"

    gpu_type = "" if compute_config["gpu_type"] is None else f":{compute_config['gpu_type']}"
    gpu_cmd = ""
    multi_gpu = compute_config["gpus"] > 1
    if compute_config["gpus"] > 0:
        gpu_cmd += " -p gpu"
        gpu_cmd += f" --gres gpu{gpu_type}:{compute_config['gpus']}"
        cmd += gpu_cmd

    cmd += f" --job-name={args.name}"

    log_dir = os.path.join(args.directory, args.name, "slurm")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cmd += f" --output={log_dir}/slurm-%j.out"

    logging.info(f"[COMPUTE] name:  {args.name}")
    logging.info(f"[COMPUTE] time:  {compute_config['time']}:00:00")
    logging.info(f"[COMPUTE] cores: {compute_config['cores']}")
    logging.info(f"[COMPUTE] total mem:   {memory}000")
    logging.info(f"[COMPUTE] gpu:   {gpu_cmd}")

    return cmd, multi_gpu


def create_configurations(config_path: Path, args: argparse.Namespace) -> tuple[list[dict], dict]:
    """
    Create individual run configurations based the provided yaml file

    Parameter
    ---------
    config_path: Path
        path to a yaml configuration for a parameter search
    args: argparse.Namespace
        command line arguments
    """

    with open(config_path, "r") as stream:
        try:
            hp_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)

    base_config = {}
    params = hp_config["params"]
    search_params = []

    for k, v in params.items():
        if type(v) == list:
            search_params.append(k)
        else:
            base_config[k] = v

    # global experiment settings
    for k in EXPERIMENT_GLOBALS:
        base_config[k] = hp_config[k]

    logging.info(f"[CONFIG] {len(search_params)} search parameters")
    logging.info(f"[CONFIG] {search_params}")

    configurations = [base_config]
    for p in search_params:
        new_configurations = []
        for v, c in product(params[p], configurations):
            new_c = c.copy()
            if type(v) == list:
                new_c[p] = " ".join(map(str, v))
            else:
                new_c[p] = v
            new_configurations.append(new_c)

        configurations = new_configurations

    return configurations, hp_config


def create_gin(config: dict, args: argparse.Namespace, run_directory: Path) -> Path:
    """
    Create a run specific gin configuration file

    Parameter
    ---------
    config: dict
        dictionary with the run specific configuration details
    args: argparse.Namespace
        command line arguments
    run_directory: Path
        run specific directory
    """

    # read base gin config
    base_config_f = open(config["base_gin"], "r")
    base_config = base_config_f.read()
    base_config_f.close()

    # create new config by adding search parameters
    final_config_path = os.path.join(run_directory, "launch_config.gin")
    with open(final_config_path, "w") as f:

        # write base config
        f.write(base_config)
        f.write("\n\n")
        f.write("#" + 50 * "=" + "\n")
        f.write("# SWEEP CONFIG\n")
        f.write("#" + 50 * "=" + "\n")

        # add options
        for h_key, h_value in sorted(config.items()):

            if h_key in EXPERIMENT_GLOBALS:
                continue

            if isinstance(h_value, str) and h_value[0] != "@":
                f.write(f"{h_key} = '{h_value}'\n")
            else:
                f.write(f"{h_key} = {h_value}\n")

        # set task
        f.write(f"ICUVariableLengthLoaderTables.task = '{config['task']}'\n")

    return Path(final_config_path)


def create_run_dir_and_gin(config: dict, args: argparse.Namespace) -> tuple[Path, Optional[Path]]:
    """
    Create the run specific directory and the associated
    gin configuration file

    Parameter
    ---------
    config: dict
        the run specific configuration
    args: argparse.Namespace
        command line arguments

    """
    run_name = generate_slug(3)
    full_run_directory = Path(args.directory / args.name / run_name)

    gin_config_path = None
    if not args.print:
        os.makedirs(full_run_directory, exist_ok=True)
        gin_config_path = create_gin(config, args, full_run_directory)

    return full_run_directory, gin_config_path


def config_to_cmd_str(
    config: dict, args: argparse.Namespace, multi_gpu_flag: bool = False
) -> list[str]:
    """
    Compute the launch command for a specific configuration and split

    Parameter
    ---------
    config: dict
        the run specific configuration
    args: argparse.Namespace
        command line arguments
    multi_gpu_flag: bool
        whether this is a multi gpu run
    """
    run_cmds = []

    # Create custom GIN (same for each seed)
    run_directory, gin_config_path = create_run_dir_and_gin(config, args)

    seed_args = [f"--seed {s}" for s in config["seeds"]]
    for seed_arg in seed_args:

        # accelerate base launch command
        run_cmd = ""
        run_cmd += f"{config['run_command']} "

        if "wandb_project" in config and config["wandb_project"] is not None:
            run_cmd += f"--wandb_project {config['wandb_project']} "

        run_cmd += f"-g {gin_config_path} "
        run_cmd += f"-l {run_directory} "
        run_cmd += seed_arg

        run_cmds.append(run_cmd)

    return run_cmds


def run_on_slurm(
    config: dict, slurm_command: str, args: argparse.Namespace, multi_gpu_flag: bool = False
) -> int:
    """
    Submit a specific run configuration to the slurm batch system

    Parameter
    ---------
    config: dict
        the run specific configuration
    slurm_command: str
        the slurm submission command based on a
        provided compute resource configuration
    args: argparse.Namespace
        command line arguments
    multi_gpu_flag: bool
        whether this is a multi-gpu run
    """
    # Check if SLURM is available
    slurm_avail = shutil.which("sbatch") is not None
    if not slurm_avail:
        logging.warning("[SUBMIT] `sbatch` not available")
        logging.warning(f"[SUBMIT] assuming local run without Slurm")

    config_cmds = config_to_cmd_str(config, args, multi_gpu_flag=multi_gpu_flag)
    for config_cmd in config_cmds:

        if slurm_avail:
            run_cmd = ' --wrap="'
            run_cmd += config_cmd
            run_cmd += '"'

            full_cmd = slurm_command + run_cmd
            # logging.info(f"Run: {full_cmd}")
        else:
            logging.info(f"[SUBMIT: slurm not available] {config_cmd}")
            full_cmd = config_cmd

        completed_process = subprocess.run(
            shlex.split(full_cmd), stdout=subprocess.DEVNULL, check=True
        )
        if completed_process.returncode != 0:
            logging.error(
                f"[SUBMIT] issue during submission, returncode: {completed_process.returncode}"
            )

    return True


# ========================
# MAIN
# ========================
def main():
    """Training Script procedure"""

    # Parse CMD arguments
    args = parse_arguments()

    logging.info("[CONFIG] Creating configurations")
    configurations, raw_config = create_configurations(args.config, args)
    logging.info(f"[CONFIG] Composed {len(configurations)} configs")

    # Check experiment directory for existing results
    experiment_directory = os.path.join(args.directory, args.name)
    if os.path.isdir(experiment_directory):
        logging.info(f"[DIRECTORY] Experiment directory: {experiment_directory} already exists")

        clear_directory = False
        if args.clear_directory:
            clear_directory = True
        elif not args.silent:
            user_input = input("[DIRECTORY] Clear experiment directory ? True / False: ")
            clear_directory = user_input.capitalize() == "True"

        if clear_directory:
            logging.warning(f"[DIRECTORY] Clearing existing experiment directory.")
            shutil.rmtree(experiment_directory)

    # Generate slurm submit command
    compute_config = {} if "compute" not in raw_config else raw_config["compute"]
    slurm_cmd, multi_gpu_flag = create_slurm_command(args, compute_config)

    # shuffle list
    if args.shuffle:
        logging.info(f"[CONFIG] shuffle configurations")
        random.shuffle(configurations)

    # Limit configs and shuffle
    if args.num_runs > 0:
        logging.info(f"[CONFIG] cut run dict to max: {args.num_runs}")
        configurations = configurations[: args.num_runs]

    # shuffle to list or dummy print
    logging.info(f"[SLURM] Submitting {len(configurations)} runs with name: {args.name}")
    for config in tqdm(configurations):
        if args.print:
            logging.warning(f"[EXEC] printing only")
            logging.info(config_to_cmd_str(config, args)[0])
        else:
            run_on_slurm(config, slurm_cmd, args, multi_gpu_flag=multi_gpu_flag)

        # to avoid race-conditions in file creation
        time.sleep(args.sleep)

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s | %(message)s")
    coloredlogs.install(level=LOGGING_LEVEL)

    # run script
    main()
