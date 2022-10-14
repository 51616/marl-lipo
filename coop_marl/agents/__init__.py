from coop_marl.agents.agent import Agent

from coop_marl.agents.qmix import QMIXAgent
from coop_marl.agents.incompat_mappo_z import IncompatMAPPOZ
from coop_marl.agents.mappo_trajedi import MAPPOTrajeDiAgent
from coop_marl.agents.mappo_rl2 import MAPPORL2Agent

__all__ = ['Agent',
            'QMIXAgent',
            'IncompatMAPPOZ',
            'MAPPOTrajeDiAgent',
            'MAPPORL2Agent']

registered_agents = {a:eval(a) for a in __all__} # dict([(a,eval(a)) for a in __all__])
