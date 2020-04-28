from copy import deepcopy
import torch
import math
from const import *
from infcomp_helper import separate_name
from pdfa.setup import generate_name_pdfa
from neural_net.dae.denoiser import SequenceDenoisingModel
from pdfa.symbol import *


class BeamInfo():
    def __init__(self, name_parser, pdfa, name):
        self.name = name
        self.log_prob = torch.Tensor([0]).to(DEVICE)
        self.encoder_info = None
        self.predictor_hidden_state = None
        self.decoder_hidden_state = None
        self.name_parser = name_parser
        self.guide_format = name_parser.guide_format
        self.pdfa = pdfa
        self.pdfa_symbols = []
        self.name_parse = None
        """
        self.component = {
            TITLE: {
                'nn': name_parser.guide_title,
                'error': [],
                'result': []
            },
            FIRST: {
                'nn': name_parser.guide_first,
                'error': [],
                'result': []
            },
            MIDDLE: {
                'nn': name_parser.guide_middle,
                'error': [],
                'result': []
            },
            LAST: {
                'nn': name_parser.guide_last,
                'error': [],
                'result': []
            },
            SUFFIX: {
                'nn': name_parser.guide_suffix,
                'error': [],
                'result': []
            }
        }
        """

    def result(self):
        components = {
            TITLE: [],
            FIRST: [],
            MIDDLE: [],
            LAST: [],
            SUFFIX: []
        }
        for sym in [TITLE, FIRST, MIDDLE, LAST, SUFFIX]:
            for parse in self.name_parse[sym]:
                components[sym].append(
                    ''.join(list(map(lambda x: PRINTABLE[x], parse))))
        result = {
            'firstname': ' '.join(components[FIRST]),
            'middlename': ' '.join(components[MIDDLE]),
            'lastname': ' '.join(components[LAST]),
            'title': ' '.join(components[TITLE]),
            'suffix': ' '.join(components[SUFFIX]),
            'log_prob': self.log_prob
        }
        return result

    def copy(self):
        copied = BeamInfo(self.name_parser, self.pdfa.copy(
            copy_state_pointer=True), self.name)
        copied.log_prob = self.log_prob.clone()
        copied.encoder_info = self.encoder_info
        copied.predictor_hidden_state = (
            self.predictor_hidden_state[0].clone(), self.predictor_hidden_state[1].clone())
        copied.name_parse = list(
            self.name_parse) if self.name_parse is not None else None
        copied.guide_format = self.guide_format
        copied.pdfa_symbols = list(self.pdfa_symbols)
        return copied


class BeamManager():
    def __init__(self, beams, beam_width):
        self.beams = beams
        self.beam_width = beam_width
        self.tmp = []

    def queue(self, candidate_beam):
        # Queue potential candidate beams
        self.tmp.append(candidate_beam)

    def select_top_k(self):
        # Only retain top k beams
        self.beams = sorted(self.tmp, key=lambda b: -
                            b.log_prob)[:self.beam_width]
        self.tmp = []

    def name_parse(self):
        # Parses the input name in all beams
        for b in self.beams:
            b.name_parse = separate_name(b.pdfa_symbols[1:], b.name)

    def result(self):
        beam_results = [b.result() for b in self.beams]
        return beam_results


def encode(neural_net, name_parse, symbol):
    encoder_outputs, encoder_hiddens = [], []
    for component in name_parse[SYMBOL]:
        output, hidden = neural_net.encode(component)
        encoder_outputs.append(output)
        encoder_hiddens.append(hidden)
    return encoder_outputs, encoder_hiddens


def beam_search(name_parser, full_name, beam_width):
    """
    Maintain top k trace information, each containing
    - Trace's PDFA
    - Trace's respective neural networks
    - Trace's Symbols
    - Trace's Total Log Probability
    ...
    """
    with torch.no_grad():
        initial_beam = BeamInfo(name_parser=name_parser,
                                pdfa=generate_name_pdfa(), name=full_name)
        bm = BeamManager([initial_beam], beam_width)
        beam_search_pdfa(bm)
        bm.name_parse()
        parsed_names_lst = bm.result()
        rank_1 = parsed_names_lst[0]
        for entry in parsed_names_lst:
            firsts = entry['firstname'].split()
            middles = entry['middlename'].split()
            lasts = entry['lastname'].split()

            top_k_firsts = dae_beam_search(name_parser.guide_fn, firsts, beam_width)
            top_k_middles = dae_beam_search(name_parser.guide_mn, middles, beam_width)
            top_k_lasts = dae_beam_search(name_parser.guide_ln, lasts, beam_width)
            print('test')

    return bm.result()


def dae_beam_search(dae: SequenceDenoisingModel, names: list, beam_width: int):
    cleaned_names = []
    for name in names:
        length_tensor = torch.LongTensor([len(name)]).to(DEVICE)
        input_tensor = torch.LongTensor(
            [dae.input.index(c) for c in name]).to(DEVICE)
        _, hidden = dae.encode(input_tensor)
        top_k = top_k_beam_search(dae, length_tensor, hidden, k=beam_width)
        cleaned_names.append(top_k)
    return cleaned_names


def top_k_beam_search(dae: SequenceDenoisingModel, length_tensor: torch.Tensor, hidden: torch.Tensor, k: int, top_penalty: float = 0.0):
    input = SOS
    output_sz = dae.decoder_output_sz
    output_chars = dae.decoder_output
    output, hidden = dae.decode(input, length_tensor, hidden)
    probs = output.reshape(output_sz)
    EOS_idx = output_chars.index(EOS)
    probs[EOS_idx] = 0
    top_k_probs, top_k_idx = torch.topk(probs, k, dim=0)

    top_k = []
    prev_chars = []
    for i in range(len(top_k_idx)):
        prev_char = output_chars[top_k_idx[i].item()]

        if i == 0:
            top_k.append(
                ([prev_char], -math.log(top_k_probs[i].item()) + top_penalty, hidden))
        else:
            top_k.append(
                ([prev_char], -math.log(top_k_probs[i].item()), hidden))

        prev_chars.append(prev_char)

    while EOS not in prev_chars:
        prev_chars = []
        hypotheses = []

        for name, score, hidden in top_k:
            prev_char = name[-1]
            input = prev_char
            output, hidden = dae.decode(input, length_tensor, hidden)
            probs = output.reshape(output_sz)
            top_k_probs, top_k_idx = torch.topk(probs, k, dim=0)
            top_k_probs = top_k_probs.reshape(k)
            top_k_idx = top_k_idx.reshape(k)

            for i in range(len(top_k_probs)):
                current_char = output_chars[top_k_idx[i]]
                current_prob = top_k_probs[i]
                name_copy = name.copy()
                name_copy.append(current_char)
                new_score = score + -math.log(current_prob)
                hypotheses.append((name_copy, new_score, hidden))

        hypotheses.sort(key=lambda x: x[1])
        top_k = hypotheses[:k]
        top_k[0] = top_k[0][0], top_k[0][1] + top_penalty, top_k[0][2]
        prev_chars = [name[-1] for name, probs, hidden in top_k]

    return top_k


def beam_search_pdfa(bm):
    """
    Beam Search Through PDFA
    """
    for b in bm.beams:
        b.encoder_info = b.guide_format.encode(b.name)
        b.predictor_hidden_state = b.guide_format.init_predictor_hidden()
        b.pdfa_symbols.append(SOS_FORMAT)

    for i in range(MAX_STRING_LEN):
        for b in bm.beams:
            if not b.pdfa.at_absorbing_state():
                emission_probs, hidden_state = b.guide_format.predict(
                    b.pdfa_symbols[-1], b.encoder_info[0][i], b.predictor_hidden_state)
                emission_probs = emission_probs * \
                    torch.Tensor(b.pdfa.get_valid_emission_mask()).to(DEVICE)
                corrected_emission_probs = emission_probs / \
                    torch.sum(emission_probs)
                for p_index, p in enumerate(corrected_emission_probs.squeeze()):
                    if p == 0:
                        continue
                    # Consider also checking whether the cumulative probability is less than least likely beam
                    # in BeamManager and reject if that is the case
                    candidate_beam = b.copy()
                    candidate_beam.pdfa_symbols.append(SYMBOL[p_index])
                    candidate_beam.log_prob += p.log()
                    candidate_beam.predictor_hidden_state = hidden_state
                    candidate_beam.pdfa.transition(SYMBOL[p_index])
                    bm.queue(candidate_beam)
            else:
                candidate_beam = b.copy()
                candidate_beam.pdfa_symbols.append(SYMBOL.index(PAD_FORMAT))
                bm.queue(candidate_beam)
        bm.select_top_k()


"""
def beam_search_denoise(bm):
    # Beam Search Through Denoising Process
    bm.name_parse()
    # iterate over each characters and feed parsed name formats into respective encoders
    for b in bm.beams:
        for sym in [TITLE, FIRST, MIDDLE, LAST, SUFFIX]:
            if sym in b.pdfa_symbols:
                outputs, hiddens = encode(b.component[sym]['nn'], b.name_parse, sym)
                b.component[sym]['encoder_outputs'] = outputs
                b.component[sym]['encoder_hiddens'] = hiddens

    for i in range(MAX_STRING_LEN):
        # iterate over each characters of beam name_format and call appropriate denoisers
        for b in bm.beams:
            for sym in [TITLE, FIRST, MIDDLE, LAST, SUFFIX]:
                curr_symbol = b.pdfa_symbols[i]
                b.component[]
"""
