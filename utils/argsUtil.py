import argparse
import os
from os import path

import yaml
from fedml import mlops


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )

    # default arguments
    parser.add_argument("--run_id", type=str, default="0")

    # default arguments
    parser.add_argument("--rank", type=int, default=0)

    # default arguments
    parser.add_argument("--local_rank", type=int, default=0)

    # For hierarchical scenario
    parser.add_argument("--node_rank", type=int, default=0)

    # default arguments
    parser.add_argument("--role", type=str, default="client")

    # default arguments
    parser.add_argument("--run_device_id", type=str, default="0")

    # default arguments
    parser.add_argument("--using_mlops", type=bool, default=False)

    # 自定义 arguments
    parser.add_argument("--run_name", type=str, default="FedAvg")

    parser.add_argument("--defense_type", type=str, default="Krum")

    parser.add_argument("--attack_rate", type=float, default=0.1)

    parser.add_argument("--enable_defense", type=bool, default=False)

    parser.add_argument("--random_attack", type=bool, default=True)


    args, unknown = parser.parse_known_args()

    if args.run_device_id != "0":
        setattr(args, "edge_id", args.run_device_id)

    return args


class Arguments:

    def __init__(self, cmd_args, training_type=None, comm_backend=None, override_cmd_args=True):
        # set the command line arguments
        cmd_args_dict = cmd_args.__dict__
        for arg_key, arg_val in cmd_args_dict.items():
            setattr(self, arg_key, arg_val)

        self.get_default_yaml_config(cmd_args, training_type, comm_backend)
        if not override_cmd_args:
            # reload cmd args again
            for arg_key, arg_val in cmd_args_dict.items():
                setattr(self, arg_key, arg_val)

    def load_yaml_config(self, yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as stream:
                try:
                    return yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise ValueError("Yaml error - check yaml file")
        except Exception as e:
            return None

    def get_default_yaml_config(self, cmd_args, training_type=None, comm_backend=None):
        if cmd_args.yaml_config_file == "":
            path_current_file = path.abspath(path.dirname(__file__))
            config_file = path.join(path_current_file, "config/simulation_sp/fedml_config.yaml")
            cmd_args.yaml_config_file = config_file
        self.yaml_paths = [cmd_args.yaml_config_file]
        configuration = self.load_yaml_config(cmd_args.yaml_config_file)
        # Override class attributes from current yaml config
        if configuration is not None:
            self.set_attr_from_config(configuration)
        if hasattr(self, "training_type"):
            training_type = self.training_type
        return configuration

    def set_attr_from_config(self, configuration):
        for _, param_family in configuration.items():
            for key, val in param_family.items():
                setattr(self, key, val)


def load_arguments(training_type=None, comm_backend=None):
    cmd_args = add_args()
    # Load all arguments from YAML config file
    args = Arguments(cmd_args, training_type, comm_backend)
    if hasattr(args, "data_cache_dir"):
        args.data_cache_dir = os.path.expanduser(args.data_cache_dir)
    if hasattr(args, "data_file_path"):
        args.data_file_path = os.path.expanduser(args.data_file_path)
    if hasattr(args, "partition_file_path"):
        args.partition_file_path = os.path.expanduser(args.partition_file_path)
    if hasattr(args, "part_file"):
        args.part_file = os.path.expanduser(args.part_file)

    args.rank = int(args.rank)
    mlops.pre_setup(args)
    if hasattr(args,"enable_tracking") and args.enable_tracking:
        if hasattr(args, "enable_wandb") and args.enable_wandb:
            wandb_only_server = getattr(args, "wandb_only_server", None)
            if (wandb_only_server and args.rank == 0 and args.process_id == 0) or not wandb_only_server:
                wandb_entity = getattr(args, "wandb_entity", None)
                if wandb_entity is not None:
                    wandb_args = {
                        "entity": args.wandb_entity,
                        "project": args.wandb_project,
                        "config": args,
                    }
                else:
                    wandb_args = {
                        "project": args.wandb_project,
                        "config": args,
                    }
                if hasattr(args, "run_name"):
                    wandb_args["name"] = args.run_name
                if hasattr(args, "wandb_group_id"):
                    wandb_args["group"] = args.wandb_group_id
                import wandb
                # wandb login
                wandb.login()
                wandb.init(**wandb_args)
    return args
