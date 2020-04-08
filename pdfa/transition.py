from .state import State


class Transition():
    """
    A class that maps a state/symbol pair to another state object

    Inputs
    - names_to_states: A dictionary that maps a unique state name to its corresponding state object
    - transition_rules: A dictionary that maps a state name/symbol tuple to another state name
    """

    def __init__(self, names_to_states: dict, transition_rules: dict):
        self.names_to_states = names_to_states
        self.transition_rules = transition_rules

    def can_transition(self, state: State, symbol: str) -> bool:
        return (state.name, symbol) in self.transition_rules

    def transition(self, state: State, symbol: str) -> State:
        new_state_name = self.transition_rules[(state.name, symbol)]
        return self.names_to_states[new_state_name]
