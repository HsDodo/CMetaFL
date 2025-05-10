import logging
import torch

def is_torch_device_available(args, device_type):
    if device_type == "gpu":
        if torch.cuda.is_available():
            return True
        return False


def get_device_type(args):
    if hasattr(args, "device_type"):
        if args.device_type == "cpu":
            device_type = "cpu"
        elif args.device_type == "gpu":
            if is_torch_device_available(args, args.device_type):
                device_type = "gpu"
            else:
                print("ML engine install was not built with GPU enabled")
                device_type = "cpu"
        else:
            raise Exception("do not support device type = {}".format(args.device_type))
    else:
        if args.using_gpu:
            if is_torch_device_available(args, "gpu"):
                device_type = "gpu"
            else:
                print("ML engine install was not built with GPU enabled")
                device_type = "cpu"
        else:
            device_type = "cpu"
    return device_type

def get_torch_device(args, using_gpu, device_id, device_type):
    logging.info(
        "args = {}, using_gpu = {}, device_id = {}, device_type = {}".format(args, using_gpu, device_id, device_type)
    )
    if using_gpu:
        gpu_id = device_id if device_id is not None else args.local_rank

        if torch.cuda.is_available() and device_type == "gpu":
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(int(gpu_id))
        else:
            device = torch.device("cpu")

        return device
    else:
        return torch.device("cpu")


def get_device(args):
    using_gpu = True if (hasattr(args, "using_gpu") and args.using_gpu is True) else False
    if args.training_type == "simulation" and args.backend == "sp":
        if not hasattr(args, "gpu_id"):
            args.gpu_id = 0
        device_type = get_device_type(args)
        device = get_torch_device(args,using_gpu, args.gpu_id, device_type)
        logging.info("device = {}".format(device))
        return device
    else:
        raise Exception(
            "the training type {} is not defined!".format(args.training_type)
        )


