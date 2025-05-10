import logging

import torch

from .models import LSTM_UCI_HAR,LSTM_UCI_HAR2, ResNet_UCI
from .models import CNN_MNIST
from .models import CNN_FeMNIST
from .models import LSTM_Shakespeare
# from .models import LSTM_PAMAP2, ResNet_PAMAP2
def create(args, output_dim):
    global model
    model_name = args.model
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    if model_name == "cnn" and args.dataset == "mnist":
        logging.info("CNN_MNIST + MNIST")
        model = CNN_MNIST(False)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FEMNIST")
        # model = CNN_DropOut(False)
        model = CNN_FeMNIST(False)
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = LSTM_Shakespeare()
    elif model_name == "LSTM" and args.dataset == "UCI-HAR":
        logging.info("LSTM + UCI-HAR")
        model = LSTM_UCI_HAR()
    elif model_name == "LSTM2" and args.dataset == "UCI-HAR":
        logging.info("LSTM + UCI-HAR")
        model = LSTM_UCI_HAR2()
    elif model_name == "ResNet" and args.dataset == "UCI-HAR":
        logging.info("ResNet + UCI-HAR")
        model = ResNet_UCI()
    else:
        raise Exception("no such model definition, please check the argument spelling or customize your own model")
    return model
