import os
from os.path import expanduser

from fedml.core import ClientTrainer, ServerAggregator, FedMLAlgorithmFlow


class Runner:
    def __init__(
        self,
        args,
        device,
        dataset,
        model,
        client_trainer: ClientTrainer = None,
        server_aggregator: ServerAggregator = None,
    ):


        if args.training_type == "simulation":
            init_runner_func = self._init_simulation_runner
        else:
            raise Exception("no such setting")

        self.runner = init_runner_func(
            args, device, dataset, model, client_trainer, server_aggregator
        )

    def _init_simulation_runner(
        self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):
        if hasattr(args, "backend") and args.backend == "sp":
            from .simulator import SimulatorSingleProcess

            runner = SimulatorSingleProcess(
                args, device, dataset, model, client_trainer, server_aggregator
            )
        else:
            raise Exception("not such backend {}".format(args.backend))
        return runner



    @staticmethod
    def log_runner_result():
        log_runner_result_dir = os.path.join(expanduser("~"), "trace")
        if not os.path.exists(log_runner_result_dir):
            os.makedirs(log_runner_result_dir, exist_ok=True)

        log_file_obj = open(os.path.join(log_runner_result_dir, str(os.getpid())), "w")
        log_file_obj.write("{}".format(str(os.getpid())))
        log_file_obj.close()

    def run(self):
        self.runner.run()
        Runner.log_runner_result()

