import torch

from .state import State
from .symbol import *
from .transition import Transition


class PDFA():
    def __init__(self,
                 name: str,
                 start_state_name: str,
                 delta: Transition,
                 device: torch.device = None,
                 outbound_symbols_to_probs: dict = None):
        self.name = name
        self.curr_state = delta.names_to_states[start_state_name]
        self.delta = delta
        self.device = device
        self.fill_incomplete_emission_probs(outbound_symbols_to_probs)
        self.outbound_symbols_to_probs = outbound_symbols_to_probs
        self.start_state_name = start_state_name

    def at_absorbing_state(self) -> bool:
        if isinstance(self.curr_state, PDFA):
            return self.curr_state.at_absorbing_state()
        return self.curr_state.absorbing

    def copy(self, copy_state_pointer: bool = False):
        """
        Copy this PDFA object excluding for its current state pointer
        """
        names_to_states_copy = {}
        for name, state in self.delta.names_to_states.items():
            if isinstance(state, PDFA):
                names_to_states_copy[name] = state.copy(copy_state_pointer)
            else:
                names_to_states_copy[name] = state
        delta_copy = Transition(names_to_states_copy, self.delta.transition_rules)
        state_pointer = self.curr_state.name if copy_state_pointer else self.start_state_name
        return PDFA(self.name, state_pointer, delta_copy, self.device, self.outbound_symbols_to_probs)

    def fill_incomplete_emission_probs(self, outbound_symbols_to_probs: dict):
        """
        Fill emission probabilities of non-top-level PDFA states based on
        outbound_symbols_to_probs dictionary that contains information on
        the likelihood of symbols that will cause the sub-PDFA
        to exit.
        """
        for state in self.delta.names_to_states.values():
            if isinstance(state, PDFA):
                state.fill_incomplete_emission_probs(outbound_symbols_to_probs)
            else:
                if outbound_symbols_to_probs is None or state.complete == True: continue
                state.set_missing_emission_probs(outbound_symbols_to_probs, normalize=True)

    def get_current_state(self) -> State:
        """
        Retrieve the current state of the PDFA. If current state is
        another PDFA, return that PDFA's current state.
        """
        if isinstance(self.curr_state, PDFA):
            return self.curr_state.get_current_state()
        return self.curr_state

    def get_emission_probs(self) -> list:
        """
        Retrieves the emission probabilities of the current state.
        If the emission probabilities are incomplete (in sub-PDFAs),
        appropriately fill them.
        """
        if isinstance(self.curr_state, PDFA):
            return self.curr_state.get_emission_probs()
        return self.curr_state.emission_probs

    def get_valid_emission_mask(self) -> list:
        """
        Retrieve an array that has a 1. for indexes of symbols that
        can be emitted from the state and 0. for others.
        """
        if isinstance(self.curr_state, PDFA):
            return self.curr_state.get_valid_emission_mask()
        emission_mask = []
        for sym in SYMBOL:
            if self.curr_state.can_emit(sym):
                emission_mask.append(1.)
            else:
                emission_mask.append(0.)
        return emission_mask

    def transition(self, symbol) -> bool:
        """
        Given a symbol, transition to next state based on self.delta.
        Return whether the transition was successful or not.
        """
        if isinstance(self.curr_state, PDFA):
            # If current state is another PDFA, call transition on it.
            # If transition was not successful (no rule in sub-PDFA),
            # we exit the sub-PDFA.
            success = self.curr_state.transition(symbol)
            if success: return True

        if self.delta.can_transition(self.curr_state, symbol):
            next_state = self.delta.transition(self.curr_state, symbol)
            self.curr_state = next_state
            return True

        return False
