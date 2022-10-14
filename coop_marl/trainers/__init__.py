from coop_marl.trainers.trainer import Trainer, trainer_setup, population_based_setup, population_evaluation, collect_data
from coop_marl.trainers.simple import SimplePSTrainer
from coop_marl.trainers.incompat import IncompatTrainer
from coop_marl.trainers.trajedi import TrajeDiTrainer
from coop_marl.trainers.meta import MetaTrainer

registered_trainers = {}

registered_trainers['simple'] = SimplePSTrainer
registered_trainers['incompat'] = IncompatTrainer
registered_trainers['trajedi'] = TrajeDiTrainer
registered_trainers['meta'] = MetaTrainer
