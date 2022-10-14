from coop_marl.envs.gym_maker import GymMaker
from coop_marl.envs.mpe.rendezvous import Rendezvous
from coop_marl.envs.overcooked.overcooked_maker import OvercookedMaker
from coop_marl.envs.one_step_matrix import OneStepMatrixGame

registered_envs = {}
registered_envs['gym_maker'] = GymMaker.make_env
registered_envs['rendezvous'] = Rendezvous.make_env
registered_envs['overcooked'] = OvercookedMaker.make_env
registered_envs['one_step_matrix'] = OneStepMatrixGame.make_env
