import pyro
import torch

from state import State
from symbol import *
from transition import Transition

class PDFA():
    def __init__(self, start_state: State, delta: Transition, device: torch.device = None):
        self.curr_state = start_state
        self.delta = delta
        self.device = device
    
    def emit(self, pyro_address: str, emission_probs=None):
        """
        If emission_probs is supplied (in q), sample a symbol from emission_probs.
        Else (in p), sample a symbol from self.curr_state.emission_probs.
        """
        if emission_probs is None:
            emission_probs = torch.tensor(self.curr_state.emission_probs).to(self.device)
        
        symbol_index = pyro.sample(pyro_address, pyro.distributions.Categorical(emission_probs))
        return SYMBOL[symbol_index]

    def transition(self, symbol) -> State:
        """
        Given a symbol, return a State object based on self.delta
        """
        next_state = self.delta.transition(self.curr_state, symbol)
        self.curr_state = next_state
        return next_state
