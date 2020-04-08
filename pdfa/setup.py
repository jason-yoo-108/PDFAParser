import re
from copy import deepcopy

from .pdfa import PDFA
from .state import State
from .symbol import *
from .transition import Transition

SPACE_DIST = [0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3]
TITLE_DIST = [0., 0.8, 0.15, 0., 0.05]
FIRST_DIST = [0.01, 0.03, 0.15, 0.2, 0.2, 0.15, 0.1, 0.08, 0.04, 0.04]
MIDDLE_DIST = [0.01, 0.03, 0.15, 0.2, 0.2, 0.15, 0.1, 0.08, 0.04, 0.04]
LAST_DIST = [0.01, 0.03, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
SUFFIX_DIST = [0.02, 0.49, 0.49]

SPACE_NAMES_TO_STATES = {
    'SPACE1': State(name='SPACE1', symbols_to_probs={}, complete=False),
    'SPACE2': State(name='SPACE2', symbols_to_probs={}, complete=False),
    'SPACE3': State(name='SPACE3', symbols_to_probs={}, complete=False),
    'SPACE4': State(name='SPACE4', symbols_to_probs={}, complete=False),
}
SPACE_TRANSITION = {
    ('SPACE1', SPACE): 'SPACE2',
    ('SPACE2', SPACE): 'SPACE3',
    ('SPACE3', SPACE): 'SPACE4',
}
TITLE_NAMES_TO_STATES = {
    'T1': State(name='T1', symbols_to_probs={}, complete=False),
    'T2': State(name='T2', symbols_to_probs={}, complete=False),
    'T3': State(name='T3', symbols_to_probs={}, complete=False),
    'T4': State(name='T4', symbols_to_probs={}, complete=False),
    'T5': State(name='T5', symbols_to_probs={}, complete=False),
}
TITLE_TRANSITION = {
    ('T1', TITLE): 'T2',
    ('T2', TITLE): 'T3',
    ('T3', TITLE): 'T4',
    ('T4', TITLE): 'T5',
}
FIRST_NAMES_TO_STATES = {
    'F1': State(name='F1', symbols_to_probs={}, complete=False),
    'F2': State(name='F2', symbols_to_probs={}, complete=False),
    'F3': State(name='F3', symbols_to_probs={}, complete=False),
    'F4': State(name='F4', symbols_to_probs={}, complete=False),
    'F5': State(name='F5', symbols_to_probs={}, complete=False),
    'F6': State(name='F6', symbols_to_probs={}, complete=False),
    'F7': State(name='F7', symbols_to_probs={}, complete=False),
    'F8': State(name='F8', symbols_to_probs={}, complete=False),
    'F9': State(name='F9', symbols_to_probs={}, complete=False),
    'F10': State(name='F10', symbols_to_probs={}, complete=False),
}
FIRST_TRANSITION = {
    ('F1', FIRST): 'F2',
    ('F2', FIRST): 'F3',
    ('F3', FIRST): 'F4',
    ('F4', FIRST): 'F5',
    ('F5', FIRST): 'F6',
    ('F6', FIRST): 'F7',
    ('F7', FIRST): 'F8',
    ('F8', FIRST): 'F9',
    ('F9', FIRST): 'F10',
}
MIDDLE_NAMES_TO_STATES = {
    'M1': State(name='M1', symbols_to_probs={}, complete=False),
    'M2': State(name='M2', symbols_to_probs={}, complete=False),
    'M3': State(name='M3', symbols_to_probs={}, complete=False),
    'M4': State(name='M4', symbols_to_probs={}, complete=False),
    'M5': State(name='M5', symbols_to_probs={}, complete=False),
    'M6': State(name='M6', symbols_to_probs={}, complete=False),
    'M7': State(name='M7', symbols_to_probs={}, complete=False),
    'M8': State(name='M8', symbols_to_probs={}, complete=False),
    'M9': State(name='M9', symbols_to_probs={}, complete=False),
    'M10': State(name='M10', symbols_to_probs={}, complete=False),
}
MIDDLE_TRANSITION = {
    ('M1', MIDDLE): 'M2',
    ('M2', MIDDLE): 'M3',
    ('M3', MIDDLE): 'M4',
    ('M4', MIDDLE): 'M5',
    ('M5', MIDDLE): 'M6',
    ('M6', MIDDLE): 'M7',
    ('M7', MIDDLE): 'M8',
    ('M8', MIDDLE): 'M9',
    ('M9', MIDDLE): 'M10',
}
LAST_NAMES_TO_STATES = {
    'L1': State(name='L1', symbols_to_probs={}, complete=False),
    'L2': State(name='L2', symbols_to_probs={}, complete=False),
    'L3': State(name='L3', symbols_to_probs={}, complete=False),
    'L4': State(name='L4', symbols_to_probs={}, complete=False),
    'L5': State(name='L5', symbols_to_probs={}, complete=False),
    'L6': State(name='L6', symbols_to_probs={}, complete=False),
    'L7': State(name='L7', symbols_to_probs={}, complete=False),
    'L8': State(name='L8', symbols_to_probs={}, complete=False),
    'L9': State(name='L9', symbols_to_probs={}, complete=False),
    'L10': State(name='L10', symbols_to_probs={}, complete=False),
}
LAST_TRANSITION = {
    ('L1', LAST): 'L2',
    ('L2', LAST): 'L3',
    ('L3', LAST): 'L4',
    ('L4', LAST): 'L5',
    ('L5', LAST): 'L6',
    ('L6', LAST): 'L7',
    ('L7', LAST): 'L8',
    ('L8', LAST): 'L9',
    ('L9', LAST): 'L10',
}
SUFFIX_NAMES_TO_STATES = {
    'S1': State(name='S1', symbols_to_probs={}, complete=False),
    'S2': State(name='S2', symbols_to_probs={}, complete=False),
    'S3': State(name='S3', symbols_to_probs={}, complete=False),
}
SUFFIX_TRANSITION = {
    ('S1', SUFFIX): 'S2',
    ('S2', SUFFIX): 'S3',
}


def fill_PDFA_stay_probs(leave_probs, names_to_states, stay_symbol):
    """
    Computes and fills the probabilities of staying in the sub-PDFA at each
    of its state.
    leave_probs: The probability of leaving the sub-PDFA at i-th index.
    names_to_states: A dictionary that maps state names to state objects.
    """
    keys = sorted(names_to_states.keys(), key=lambda k: int(re.findall(r'\d+', k)[0]))
    p_current_node = 1.

    for i in range(len(keys)):
        state = names_to_states[keys[i]]
        target_leave_prob = leave_probs[i]
        curr_leave_prob = target_leave_prob / p_current_node
        curr_stay_prob = 1. - curr_leave_prob
        state.set_missing_emission_probs({stay_symbol: curr_stay_prob})
        p_current_node = p_current_node * curr_stay_prob


fill_PDFA_stay_probs(SPACE_DIST, SPACE_NAMES_TO_STATES, SPACE)
fill_PDFA_stay_probs(TITLE_DIST, TITLE_NAMES_TO_STATES, TITLE)
fill_PDFA_stay_probs(FIRST_DIST, FIRST_NAMES_TO_STATES, FIRST)
fill_PDFA_stay_probs(MIDDLE_DIST, MIDDLE_NAMES_TO_STATES, MIDDLE)
fill_PDFA_stay_probs(LAST_DIST, LAST_NAMES_TO_STATES, LAST)
fill_PDFA_stay_probs(SUFFIX_DIST, SUFFIX_NAMES_TO_STATES, SUFFIX)

NAMES_TO_STATES = {
    'START': State(name='START', symbols_to_probs={SOS_FORMAT: 1.}),
    'START_SOS': State(name='START_SOS', symbols_to_probs={FIRST: 0.4, LAST: 0.4, TITLE: 0.15, SPACE: 0.05}),
    'START_SPACE': PDFA(
        name='START_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={TITLE: 0.1, FIRST: 0.45, LAST: 0.45}
    ),
    'TITLE': PDFA(
        name='TITLE',
        start_state_name='T1',
        delta=Transition(names_to_states=deepcopy(TITLE_NAMES_TO_STATES), transition_rules=TITLE_TRANSITION),
        outbound_symbols_to_probs={PERIOD: 0.5, SPACE: 0.45, FIRST: 0.025, LAST: 0.025}
    ),
    'TITLE_P': State(name='TITLE_P', symbols_to_probs={SPACE: 0.95, FIRST: 0.025, LAST: 0.025}),
    'TITLE_SPACE': PDFA(
        name='TITLE_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={FIRST: 0.5, LAST: 0.5}
    ),
    'FIRST_FML': PDFA(
        name='FIRST_FML',
        start_state_name='F1',
        delta=Transition(names_to_states=deepcopy(FIRST_NAMES_TO_STATES), transition_rules=FIRST_TRANSITION),
        outbound_symbols_to_probs={SPACE: 0.95, MIDDLE: 0.025, LAST: 0.025}
    ),
    'FIRST_FML_SPACE': PDFA(
        name='FIRST_FML_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={MIDDLE: 0.5, LAST: 0.5}
    ),
    'MIDDLE_FML_1': PDFA(
        name='MIDDLE_FML_1',
        start_state_name='M1',
        delta=Transition(names_to_states=deepcopy(MIDDLE_NAMES_TO_STATES), transition_rules=MIDDLE_TRANSITION),
        outbound_symbols_to_probs={SPACE: 0.8, PERIOD: 0.175, LAST: 0.025}
    ),
    'MIDDLE_FML_1_P': State(name='MIDDLE_FML_1_P', symbols_to_probs={SPACE: 0.95, MIDDLE: 0.05}),
    'MIDDLE_FML_1_SPACE': PDFA(
        name='MIDDLE_FML_1_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={LAST: 0.8, MIDDLE: 0.2}
    ),
    'MIDDLE_FML_2': PDFA(
        name='MIDDLE_FML_2',
        start_state_name='M1',
        delta=Transition(names_to_states=deepcopy(MIDDLE_NAMES_TO_STATES), transition_rules=MIDDLE_TRANSITION),
        outbound_symbols_to_probs={SPACE: 0.8, PERIOD: 0.175, LAST: 0.025}
    ),
    'MIDDLE_FML_2_P': State(name='MIDDLE_FML_2_P', symbols_to_probs={SPACE: 0.95, LAST: 0.05}),
    'MIDDLE_FML_2_SPACE': PDFA(
        name='MIDDLE_FML_2_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={LAST: 1.}
    ),
    'LAST_FML': PDFA(
        name='LAST_FML',
        start_state_name='L1',
        delta=Transition(names_to_states=deepcopy(LAST_NAMES_TO_STATES), transition_rules=LAST_TRANSITION),
        outbound_symbols_to_probs={EOS_FORMAT: 0.8, SPACE: 0.175, SUFFIX: 0.025}
    ),
    'LAST_LFM': PDFA(
        name='LAST_LFM',
        start_state_name='L1',
        delta=Transition(names_to_states=deepcopy(LAST_NAMES_TO_STATES), transition_rules=LAST_TRANSITION),
        outbound_symbols_to_probs={COMMA: 0.975, SPACE: 0.025}
    ),
    'LAST_LFM_C': State(name='LAST_LFM_C', symbols_to_probs={SPACE: 0.95, FIRST: 0.05}),
    'LAST_LFM_SPACE': PDFA(
        name='LAST_LFM_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={FIRST: 1.}
    ),
    'FIRST_LFM': PDFA(
        name='FIRST_LFM',
        start_state_name='F1',
        delta=Transition(names_to_states=deepcopy(FIRST_NAMES_TO_STATES), transition_rules=FIRST_TRANSITION),
        outbound_symbols_to_probs={EOS_FORMAT: 0.5, SPACE: 0.475, MIDDLE: 0.025}
    ),
    'FIRST_LFM_SPACE': PDFA(
        name='FIRST_LFM_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={MIDDLE: 0.8, SUFFIX: 0.2}
    ),
    'MIDDLE_LFM_1': PDFA(
        name='MIDDLE_LFM_1',
        start_state_name='M1',
        delta=Transition(names_to_states=deepcopy(MIDDLE_NAMES_TO_STATES), transition_rules=MIDDLE_TRANSITION),
        outbound_symbols_to_probs={SPACE: 0.8, PERIOD: 0.175, SUFFIX: 0.025}
    ),
    'MIDDLE_LFM_1_P': State(name='MIDDLE_LFM_1_P', symbols_to_probs={SPACE: 0.95, MIDDLE: 0.05}),
    'MIDDLE_LFM_1_SPACE': PDFA(
        name='MIDDLE_LFM_1_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={SUFFIX: 0.8, MIDDLE: 0.2}
    ),
    'MIDDLE_LFM_2': PDFA(
        name='MIDDLE_LFM_2',
        start_state_name='M1',
        delta=Transition(names_to_states=deepcopy(MIDDLE_NAMES_TO_STATES), transition_rules=MIDDLE_TRANSITION),
        outbound_symbols_to_probs={EOS_FORMAT: 0.8, SPACE: 0.1, PERIOD: 0.1}
    ),
    'MIDDLE_LFM_2_P': State(name='MIDDLE_LFM_2_P', symbols_to_probs={EOS_FORMAT: 0.8, SPACE: 0.175, SUFFIX: 0.025}),
    'SUFFIX_SPACE': PDFA(
        name='SUFFIX_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={SUFFIX: 0.95, EOS_FORMAT: 0.05}
    ),
    'SUFFIX': PDFA(
        name='SUFFIX',
        start_state_name='S1',
        delta=Transition(names_to_states=deepcopy(SUFFIX_NAMES_TO_STATES), transition_rules=SUFFIX_TRANSITION),
        outbound_symbols_to_probs={EOS_FORMAT: 0.95, SPACE: 0.05}
    ),
    'END_SPACE': PDFA(
        name='END_SPACE',
        start_state_name='SPACE1',
        delta=Transition(names_to_states=deepcopy(SPACE_NAMES_TO_STATES), transition_rules=SPACE_TRANSITION),
        outbound_symbols_to_probs={EOS_FORMAT: 1.}
    ),
    'END': State(name='END', symbols_to_probs={}, complete=False, absorbing=True),
}

FULLNAME_TRANSITION_RULES = {
    ('START', SOS_FORMAT): 'START_SOS',
    ('START_SOS', FIRST): 'FIRST_FML',
    ('START_SOS', LAST): 'LAST_LFM',
    ('START_SOS', TITLE): 'TITLE',
    ('START_SOS', SPACE): 'START_SPACE',
    ('START_SPACE', FIRST): 'FIRST_FML',
    ('START_SPACE', LAST): 'LAST_LFM',
    ('START_SPACE', TITLE): 'TITLE',
    ('TITLE', PERIOD): 'TITLE_P',
    ('TITLE', SPACE): 'TITLE_SPACE',
    ('TITLE', FIRST): 'FIRST_FML',
    ('TITLE', LAST): 'LAST_LFM',
    ('TITLE_P', SPACE): 'TITLE_SPACE',
    ('TITLE_P', FIRST): 'FIRST_FML',
    ('TITLE_P', LAST): 'LAST_LFM',
    ('TITLE_SPACE', FIRST): 'FIRST_FML',
    ('TITLE_SPACE', LAST): 'LAST_LFM',
    ('FIRST_FML', SPACE): 'FIRST_FML_SPACE',
    ('FIRST_FML', MIDDLE): 'MIDDLE_FML_1',
    ('FIRST_FML', LAST): 'LAST_FML',
    ('FIRST_FML_SPACE', MIDDLE): 'MIDDLE_FML_1',
    ('FIRST_FML_SPACE', LAST): 'LAST_FML',
    ('MIDDLE_FML_1', SPACE): 'MIDDLE_FML_1_SPACE',
    ('MIDDLE_FML_1', PERIOD): 'MIDDLE_FML_1_P',
    ('MIDDLE_FML_1', LAST): 'LAST_FML',
    ('MIDDLE_FML_1_P', SPACE): 'MIDDLE_FML_1_SPACE',
    ('MIDDLE_FML_1_P', MIDDLE): 'MIDDLE_FML_2',
    ('MIDDLE_FML_1_SPACE', LAST): 'LAST_FML',
    ('MIDDLE_FML_1_SPACE', MIDDLE): 'MIDDLE_FML_2',
    ('MIDDLE_FML_2', SPACE): 'MIDDLE_FML_2_SPACE',
    ('MIDDLE_FML_2', PERIOD): 'MIDDLE_FML_2_P',
    ('MIDDLE_FML_2', LAST): 'LAST_FML',
    ('MIDDLE_FML_2_P', SPACE): 'MIDDLE_FML_2_SPACE',
    ('MIDDLE_FML_2_P', LAST): 'LAST_FML',
    ('MIDDLE_FML_2_SPACE', LAST): 'LAST_FML',
    ('LAST_FML', EOS_FORMAT): 'END',
    ('LAST_FML', SPACE): 'SUFFIX_SPACE',
    ('LAST_FML', SUFFIX): 'SUFFIX',
    ('LAST_LFM', COMMA): 'LAST_LFM_C',
    ('LAST_LFM', SPACE): 'LAST_LFM_SPACE',
    ('LAST_LFM_C', SPACE): 'LAST_LFM_SPACE',
    ('LAST_LFM_C', FIRST): 'FIRST_LFM',
    ('LAST_LFM_SPACE', FIRST): 'FIRST_LFM',
    ('FIRST_LFM', EOS_FORMAT): 'END',
    ('FIRST_LFM', SPACE): 'FIRST_LFM_SPACE',
    ('FIRST_LFM', MIDDLE): 'MIDDLE_LFM_1',
    ('FIRST_LFM_SPACE', MIDDLE): 'MIDDLE_LFM_1',
    ('FIRST_LFM_SPACE', SUFFIX): 'SUFFIX',
    ('MIDDLE_LFM_1', SPACE): 'MIDDLE_LFM_1_SPACE',
    ('MIDDLE_LFM_1', PERIOD): 'MIDDLE_LFM_1_P',
    ('MIDDLE_LFM_1', SUFFIX): 'SUFFIX',
    ('MIDDLE_LFM_1_P', SPACE): 'MIDDLE_LFM_1_SPACE',
    ('MIDDLE_LFM_1_P', MIDDLE): 'MIDDLE_LFM_2',
    ('MIDDLE_LFM_1_SPACE', SUFFIX): 'SUFFIX',
    ('MIDDLE_LFM_1_SPACE', MIDDLE): 'MIDDLE_LFM_2',
    ('MIDDLE_LFM_2', EOS_FORMAT): 'END',
    ('MIDDLE_LFM_2', SPACE): 'SUFFIX_SPACE',
    ('MIDDLE_LFM_2', PERIOD): 'MIDDLE_LFM_2_P',
    ('MIDDLE_LFM_2_P', EOS_FORMAT): 'END',
    ('MIDDLE_LFM_2_P', SPACE): 'SUFFIX_SPACE',
    ('MIDDLE_LFM_2_P', SUFFIX): 'SUFFIX',
    ('SUFFIX_SPACE', SUFFIX): 'SUFFIX',
    ('SUFFIX_SPACE', EOS_FORMAT): 'END',
    ('SUFFIX', EOS_FORMAT): 'END',
    ('SUFFIX', SPACE): 'END_SPACE',
    ('END_SPACE', EOS_FORMAT): 'END',
}

CANONICAL_PDFA = PDFA(
    name='CANONICAL',
    start_state_name='START',
    delta=Transition(
        names_to_states=NAMES_TO_STATES,
        transition_rules=FULLNAME_TRANSITION_RULES
    )
)


def generate_name_pdfa() -> PDFA:
    return deepcopy(CANONICAL_PDFA)
