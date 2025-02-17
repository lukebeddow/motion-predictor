"""
Wrappers to build Agents

"""

import policy_gradient
from trainers.dreamer_trainer import DreamerTrainer


def build_ppo(agent_cfg:dict, **kwargs:dict):

    network = getattr(policy_gradient, agent_cfg.network)
    agent = getattr(policy_gradient, agent_cfg.agent)
    agent.init(network)

    return agent

def build_dreamer(agent_cfg:dict, **kwargs:dict):

    dreamer = DreamerTrainer(agent_cfg)
    return dreamer





