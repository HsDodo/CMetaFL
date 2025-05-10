import os
import sys

sys.path.append("..")
class SimulatorSingleProcess:
    def __init__(self, args, device, dataset, model, client_trainer=None, server_aggregator=None):
        from ..Optimizer.FedAvg import FedAvg
        from ..Optimizer.CMetaFL import CMetaFL
        from ..Optimizer.fedprox_trainer import FedProxTrainer
        if args.federated_optimizer == "FedAvg":
            self.fl_trainer = FedAvg(args, device, dataset, model)
        elif args.federated_optimizer == "CMetaFL":
            self.fl_trainer = CMetaFL(args, device, dataset, model)
        elif args.federated_optimizer == "FedProx":
            self.fl_trainer = FedProxTrainer(dataset, model, device, args)
        else:
            raise Exception("Exception")

    def run(self):
        self.fl_trainer.train()

