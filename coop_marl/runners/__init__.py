from coop_marl.runners.runners import EpisodesRunner, StepsRunner

__all__ = ['EpisodesRunner', 'StepsRunner']

registered_runners = {a:eval(a) for a in __all__}
