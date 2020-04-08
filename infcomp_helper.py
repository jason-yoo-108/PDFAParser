import pyro
import pyro.distributions as dist
import torch

from neural_net.cc_model import CharacterClassificationModel
from neural_net.pretrained.name_generator import *
from noise.noise import *
from noise.noise_class import *
from pdfa.pdfa import PDFA
from pdfa.symbol import *


def sample_pdfa(pdfa: PDFA) -> list:
    name_format = []
    for i in range(MAX_STRING_LEN):
        if not pdfa.at_absorbing_state():
            emission_probs = torch.Tensor(pdfa.get_emission_probs()).to(DEVICE)
            symbol = SYMBOL[pyro.sample(f"{ADDRESS['format']}_{i}", dist.Categorical(emission_probs))]
            transition_success = pdfa.transition(symbol)
            if not transition_success: raise Exception(
                f"PDFA transition with state/symbol {pdfa.get_current_state().name}/{symbol} failed.")
        else:
            emission_probs = torch.zeros(len(SYMBOL)).to(DEVICE)
            emission_probs[SYMBOL.index(PAD_FORMAT)] = 1.
            symbol = SYMBOL[pyro.sample(f"{ADDRESS['format']}_{i}", dist.Categorical(emission_probs))]
        name_format.append(symbol)
    return name_format


def sample_conditional_pdfa(pdfa: PDFA, inference_network: CharacterClassificationModel,
                            observed: torch.Tensor) -> list:
    name_format = []
    encoder_output, encoder_hidden = inference_network.encode(observed)
    symbol, hidden_state = SOS_FORMAT, inference_network.init_predictor_hidden()

    for i in range(MAX_STRING_LEN):
        if not pdfa.at_absorbing_state():
            emission_probs, hidden_state = inference_network.predict(symbol, encoder_output[i], hidden_state)
            emission_probs = emission_probs * torch.Tensor(pdfa.get_valid_emission_mask()).to(DEVICE)
            corrected_emission_probs = emission_probs / torch.sum(emission_probs)
            if i == 0:
                # Always sample SOS at the beginning
                corrected_emission_probs = torch.Tensor(pdfa.get_emission_probs()).to(DEVICE)
            symbol = SYMBOL[pyro.sample(f"{ADDRESS['format']}_{i}", dist.Categorical(corrected_emission_probs))]
            transition_success = pdfa.transition(symbol)
            if not transition_success: raise Exception(
                f"PDFA transition with state/symbol {pdfa.get_current_state().name}/{symbol} failed.")
        else:
            emission_probs = torch.zeros(len(SYMBOL)).to(DEVICE)
            emission_probs[SYMBOL.index(PAD_FORMAT)] = 1.
            symbol = SYMBOL[pyro.sample(f"{ADDRESS['format']}_{i}", dist.Categorical(emission_probs))]
        name_format.append(symbol)
    return name_format


def separate_components(name_format: list) -> dict:
    result = {TITLE: [], FIRST: [], MIDDLE: [], LAST: [], SUFFIX: []}
    last_sym, last_sym_index = name_format[0], 0
    for i, sym in enumerate(name_format):
        if sym != last_sym:
            if last_sym in result:
                result[last_sym].append(name_format[last_sym_index:i])
            last_sym, last_sym_index = sym, i
    if name_format[-1] in list(result.keys()):
        result[last_sym].append(name_format[last_sym_index:i])
    return result


def separate_name(name_format: list, name: torch.Tensor) -> dict:
    result = {TITLE: [], FIRST: [], MIDDLE: [], LAST: [], SUFFIX: []}
    last_sym, last_sym_index = name_format[0], 0
    for i, sym in enumerate(name_format):
        if sym != last_sym:
            if last_sym in result:
                result[last_sym].append(name[last_sym_index:i])
            last_sym, last_sym_index = sym, i
    if name_format[-1] in list(result.keys()):
        result[last_sym].append(name[last_sym_index:i])
    return result


def sample_title_and_noise(title_format: list, rnn: ClassificationDenoisingModel = None, encoder_output=None,
                           encoder_hidden=None) -> tuple:
    """
    Samples a title based on title_format and noise_classes.
    Ensures title_length is either 2, 3, or 5.
    """
    title_length, noise_classes = sample_title_noise(title_format_length=len(title_format[0]), rnn=rnn,
                                                     encoder_output=encoder_output)

    if rnn is None:
        # In model
        title_probs = torch.zeros(len(TITLE_LIST)).to(DEVICE)
        if title_length == 2:
            for index, prob in TITLE_2.values(): title_probs[index] = prob
        if title_length == 3:
            for index, prob in TITLE_3.values(): title_probs[index] = prob
        if title_length == 5:
            for index, prob in TITLE_5.values(): title_probs[index] = prob
    else:
        # In guide
        title_probs = rnn.classify(encoder_hidden)
        title_mask = torch.zeros(len(TITLE_LIST)).to(DEVICE)
        if title_length == 2:
            for index, prob in TITLE_2.values(): title_mask[index] = 1.
        if title_length == 3:
            for index, prob in TITLE_3.values(): title_mask[index] = 1.
        if title_length == 5:
            for index, prob in TITLE_5.values(): title_mask[index] = 1.
        title_probs = title_probs * title_mask / torch.sum(title_probs * title_mask)

    title = TITLE_LIST[pyro.sample(f"{ADDRESS['title']}", dist.Categorical(title_probs))]
    return title, noise_classes


def sample_name_and_noise(name_format: list, rnn, pyro_address: str, encoder_output=None, encoder_hidden=None) -> tuple:
    name = []
    in_model = isinstance(rnn, NameGenerator)

    if in_model:
        name_length, noise_classes = sample_name_noise(name_format_length=len(name_format[0]),
                                                       pyro_address=pyro_address)
        rnn_input = rnn.indexTensor([[SOS]], 1).to(DEVICE)
        length_input = rnn.lengthTestTensor([[name_length]]).to(DEVICE)
        hidden = None

        for i in range(name_length):
            rnn_probs, hidden = rnn.forward(rnn_input[0], length_input, hidden)
            char_index = pyro.sample(f"{pyro_address}_{i}", dist.Categorical(rnn_probs.exp())).squeeze()
            char = rnn.output[char_index]
            name.append(char)
            rnn_input = rnn.indexTensor([[char]], 1).to(DEVICE)
    else:
        name_length, noise_classes = sample_name_noise(name_format_length=len(name_format[0]),
                                                       pyro_address=pyro_address, rnn=rnn,
                                                       encoder_output=encoder_output)

        rnn_input = SOS
        hidden = encoder_hidden
        for i in range(name_length):
            rnn_probs, hidden = rnn.decode(rnn_input, name_length, hidden)
            char_index = pyro.sample(f"{pyro_address}_{i}", dist.Categorical(rnn_probs)).squeeze()
            rnn_input = rnn.decoder_output[char_index]
            name.append(rnn_input)

    return name, noise_classes


def sample_multiple_name_and_noise(name_formats: list, rnn, pyro_address: str, encoder_outputs=None,
                                   encoder_hiddens=None) -> tuple:
    names, multiple_noise_classes = [], []
    for i, name_format in enumerate(name_formats):
        encoder_output = encoder_outputs[i] if encoder_outputs is not None else None
        encoder_hidden = encoder_hiddens[i] if encoder_hiddens is not None else None
        name, noise_classes = sample_name_and_noise([name_format], rnn, f"{pyro_address}_{i}", encoder_output,
                                                    encoder_hidden)
        names.append(name)
        multiple_noise_classes.append(noise_classes)
    return names, multiple_noise_classes


def sample_suffix_and_noise(suffix_format: list, rnn: ClassificationDenoisingModel = None, encoder_output=None,
                            encoder_hidden=None) -> tuple:
    """
    Samples a suffix based on suffix_format and noise_classes.
    """
    suffix_length, noise_classes = sample_suffix_noise(suffix_format_length=len(suffix_format[0]), rnn=rnn,
                                                       encoder_output=encoder_output)

    if rnn is None:
        # In model
        suffix_probs = torch.zeros(len(SUFFIX_LIST)).to(DEVICE)
        if suffix_length == 1:
            for index, prob in SUFFIX_1.values(): suffix_probs[index] = prob
        if suffix_length == 2:
            for index, prob in SUFFIX_2.values(): suffix_probs[index] = prob
        if suffix_length == 3:
            for index, prob in SUFFIX_3.values(): suffix_probs[index] = prob
    else:
        # In guide
        suffix_probs = rnn.classify(encoder_hidden)
        suffix_mask = torch.zeros(len(SUFFIX_LIST)).to(DEVICE)
        if suffix_length == 1:
            for index, prob in SUFFIX_1.values(): suffix_mask[index] = 1.
        if suffix_length == 2:
            for index, prob in SUFFIX_2.values(): suffix_mask[index] = 1.
        if suffix_length == 3:
            for index, prob in SUFFIX_3.values(): suffix_mask[index] = 1.
        suffix_probs = suffix_probs * suffix_mask / torch.sum(suffix_probs * suffix_mask)

    suffix = SUFFIX_LIST[pyro.sample(f"{ADDRESS['suffix']}", dist.Categorical(suffix_probs))]
    return suffix, noise_classes


def observation_probabilities(original: list, noise_classes: list, peak_prob: float) -> torch.Tensor:
    noised_tensor = []
    i = 0  # Tracks the index of the latent name for noising purposes
    # try:
    for noise_class in noise_classes:
        if i > len(original): raise Exception(
            f"Index greater than original list; info (i/original/noise_class): {i}, {original}, {noise_classes}")
        if noise_class == NOISE_NONE:
            noised_tensor.append(insert_peaked_probs(original[i], peak_prob))
            i += 1
        elif noise_class == NOISE_ADD:
            # JASON XAXX => JAON
            noised_tensor.append(insert_peaked_probs(original[i], peak_prob))
            i += 2
        elif noise_class == NOISE_REPLACE:
            # JASON XRXXX => JXSON
            noised_tensor.append(insert_noise_probs(original[i]))
            i += 1
        elif noise_class == NOISE_DELETE:
            # JASON XDXXXX => JXASON
            if i >= len(original):
                index = len(original) - 1
            else:
                index = i
            noised_tensor.append(insert_noise_probs(original[index]))

    if len(noised_tensor) != len(noise_classes):
        raise Exception(
            f"Length Mismatch Between Noised Tensor and Noise Classes: {len(noised_tensor)} vs {len(noise_classes)}")
    # except Exception as e:
    #    raise e

    return noised_tensor


def one_hot_char(self, char: str) -> list:
    result = [0] * NUM_PRINTABLE
    result[PRINTABLE.index(char)] = 1.
    return result


def combine_observation_probabilities(name_format: list, title: torch.Tensor, firstname: torch.Tensor,
                                      middlenames: list, lastname: torch.Tensor, suffix: torch.Tensor,
                                      peak_prob: int) -> torch.Tensor:
    try:
        combined_probs = []
        i_title, i_first, i_middle, i_last, i_suffix = 0, 0, 0, 0, 0
        middlename_index = 0

        for i, f in enumerate(name_format):
            if f == TITLE:
                combined_probs.append(title[i_title])
                i_title += 1
            elif f == FIRST:
                combined_probs.append(firstname[i_first])
                i_first += 1
            elif f == MIDDLE:
                if i_middle >= len(middlenames[0]):
                    middlename_index = 1
                    i_middle = 0
                combined_probs.append(middlenames[middlename_index][i_middle])
                i_middle += 1
            elif f == LAST:
                combined_probs.append(lastname[i_last])
                i_last += 1
            elif f == SUFFIX:
                combined_probs.append(suffix[i_suffix])
                i_suffix += 1
            elif f == SPACE:
                combined_probs.append(insert_peaked_probs(SPACE, peak_prob))
            elif f == COMMA:
                combined_probs.append(insert_peaked_probs(COMMA, peak_prob))
            elif f == PERIOD:
                combined_probs.append(insert_peaked_probs(PERIOD, peak_prob))
            elif f == EOS_FORMAT:
                combined_probs.append(insert_peaked_probs(EOS, peak_prob))
            else:
                combined_probs.append(insert_peaked_probs(PAD, peak_prob))
    except Exception as e:
        raise Exception(f"Exception Message: {e}; NameFormat/Title/First/Middle/Last/Suffix Shapes: " +
                        f"{name_format}/{title.shape}/{firstname.shape}/{' '.join([m.shape for m in middlenames])}/{lastname.shape}/{suffix.shape}")
    return torch.Tensor(combined_probs).to(DEVICE)
