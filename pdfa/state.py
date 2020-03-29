from pdfa import PDFA
from symbol import *

class State():
    """
    A PDFA state with a unique name and a dictionary including the probabilities
    of sampling specific symbols.

    Example: State(name='START_STATE', symbols_to_probs={TITLE: 0.1, FIRST: 0.45, LAST: 0.45})
    """
    def __init__(self, name: str, symbols_to_probs: dict):
        self.name = name
        self.symbols = list(symbols_to_probs.keys())
        self.symbols_to_probs = symbols_to_probs
        self.emission_probs = self._set_emission_probs(symbols_to_probs)
    
    def can_emit(self, symbol: str) -> bool:
        return symbol in self.symbols
    
    def emission_prob(self, symbol: str) -> float:
        return self.symbols_to_probs[symbol]

    def _set_emission_probs(self, symbols_to_probs) -> list:
        result = [0.] * len(SYMBOL)
        for symbol, prob in symbols_to_probs.items():
            result[SYMBOL.index(symbol)] = prob
        return result


class SuperState(State):
    """
    A PDFA state with a unique name and a dictionary including the probabilities
    of sampling specific symbols. It contains a smaller PDFA with its own transition
    rules. Upon initialization, a SuperState injects the missing emission probabilities 
    in its smaller PDFA states that correspond to the probability of exiting the 
    SuperState. Once the smaller PDFA samples a symbol it does not have a transition
    rule for, it returns the control flow to the SuperState.
    """
    def __init__(self, name: str, symbols_to_probs: dict, pdfa: PDFA):
        super().__init__(name, symbols_to_probs)
        self.pdfa = pdfa
    
    def emit(self, pyro_address: str, emission_probs=None) -> str:
        """
        """
        self.pdfa.emit(pyro_address, emission_probs)
        pass
