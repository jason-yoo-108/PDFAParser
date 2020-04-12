from .symbol import SYMBOL


class State():
    """
    A PDFA state with a unique name and a dictionary including the probabilities
    of sampling specific symbols.

    Example: State(name='START_STATE', symbols_to_probs={TITLE: 0.1, FIRST: 0.45, LAST: 0.45})
    """

    def __init__(self, name: str, symbols_to_probs: dict, complete: bool = True, absorbing: bool = False):
        self.name = name
        self.symbols = list(symbols_to_probs.keys())
        self.symbols_to_probs = symbols_to_probs
        self.emission_probs = self._set_emission_probs(symbols_to_probs)
        self.complete = complete
        self.absorbing = absorbing

    def can_emit(self, symbol: str) -> bool:
        return symbol in self.symbols and self.symbols_to_probs[symbol] > 1e-6

    def emission_prob(self, symbol: str) -> float:
        return self.symbols_to_probs[symbol]

    def set_missing_emission_probs(self, extra_symbols_to_probs: dict, normalize: bool = False):
        # Only call on start of the program
        # Sets (partial) probability values in extra_symbols_to_probs into self.symbols_to_probs
        # If normalize is true, scale the probs in extra_symbols_to_probs to fill the unallocated probabilities
        total_unallocated_probs = 1. - sum(self.symbols_to_probs.values())
        for symbol, prob in extra_symbols_to_probs.items():
            if normalize:
                prob = prob * total_unallocated_probs
            if symbol not in self.symbols_to_probs:
                self.symbols_to_probs[symbol] = abs(prob) if prob > 1e-6 else 0
        self.symbols = list(self.symbols_to_probs.keys())
        self.emission_probs = self._set_emission_probs(self.symbols_to_probs)

    def _set_emission_probs(self, symbols_to_probs) -> list:
        result = [0.] * len(SYMBOL)
        for symbol, prob in symbols_to_probs.items():
            result[SYMBOL.index(symbol)] = prob
        return result
