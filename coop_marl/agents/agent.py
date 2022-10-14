from coop_marl.utils import Arrdict

class Agent:
    # Every agent should have these functions
    def __init__(self, config):
        self.validate_config(config)

    def act(self, inp):
        raise NotImplementedError

    def preprocess(self, traj):
        raise NotImplementedError

    def train(self, batch):
        raise NotImplementedError

    # def get_dummy_decision(self):
    #     raise NotImplementedError

    def get_prev_decision_view(self):
        # raise NotImplementedError
        return Arrdict()

    def reset(self):
        raise NotImplementedError

    def validate_config(self, config):
        raise NotImplementedError