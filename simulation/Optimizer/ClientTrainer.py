
from abc import ABC, abstractmethod


class ClientTrainer(ABC):
    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args
        self.local_train_dataset = None
        self.local_test_dataset = None
        self.local_sample_number = 0
        self.rid = 0
        self.template_model_params = self.get_model_params()
        self.enc_model_params =  None

    def set_id(self, trainer_id):
        self.id = trainer_id

    def is_main_process(self):
        return True

    def update_dataset(self, local_train_dataset, local_test_dataset, local_sample_number):
        self.local_train_dataset = local_train_dataset
        self.local_test_dataset = local_test_dataset
        self.local_sample_number = local_sample_number

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def get_enc_model_params(self):
        return self.enc_model_params

    def set_enc_model_params(self, enc_model_parameters):
        self.enc_model_params = enc_model_parameters


    @abstractmethod
    def train(self, train_data, device, args):
        pass


    def test(self, test_data, device, args):
        pass
