import fedml

from simulation.Runner.runner import Runner
import model.model_hub as moder_hub
from utils import device
from utils import argsUtil
import data.data_loder as data_loader
if __name__ == "__main__":

    # load arguments
    args = argsUtil.load_arguments()
    # init device
    device = device.get_device(args)

    # load data
    dataset, output_dim = data_loader.load(args)

    # load model
    model = moder_hub.create(args, output_dim)
    # start training
    runner = Runner(args, device, dataset, model)
    runner.run()