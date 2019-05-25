import torch
from collections import OrderedDict


def load_troch_model(filename):
    def _process_state(state):
        new_state_dict = OrderedDict()
        for k, v in state.items():
            if k.startswith('module.'):
                k = k.replace('module.', '')
            new_state_dict[k] = v
        return new_state_dict
    state = torch.load(filename, map_location=lambda storage, location: storage)
    state['state_dict'] = _process_state(state['state_dict'])
    return state


def load_troch_model2(filename):
    def _process_state(state):
        new_state_dict = OrderedDict()
        for k, v in state.items():
            if k.startswith('module.'):
                k = k.replace('module.', '')
            new_state_dict[k] = v
        return new_state_dict
    state = torch.load(filename, map_location=lambda storage, location: storage)
    state = _process_state(state)
    return state
