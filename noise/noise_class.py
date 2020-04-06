import pyro
import pyro.distributions as dist
import torch

from const import *
from neural_net.dae.denoiser import ClassificationDenoisingModel, SequenceDenoisingModel



def forbid_noise_class(probs: torch.Tensor, forbidden: list) -> torch.Tensor:
    # Set forbidden noise class probabilities as 0 and normalize probs in-place
    mask = torch.ones(len(probs)).to(DEVICE)
    forbidden.append(NOISE_SOS)
    for noise_class in forbidden:
        mask[NOISE.index(noise_class)] = 0.
    probs = probs * mask
    return probs/torch.sum(probs)


def sample_title_noise(title_format_length, rnn: ClassificationDenoisingModel = None, encoder_output: torch.Tensor = None) -> tuple:
    title_length, noise_classes = 0, []
    if rnn is not None:
        hidden_state = rnn.init_predictor_hidden()
        noise_type = NOISE_SOS
    
    for i in range(title_format_length):
        if rnn is not None:
            # in guide
            if i > 0: noise_type = noise_classes[-1]
            char_noise_class_probs, hidden_state = rnn.predict(noise_type, encoder_output[i], hidden_state)
            char_noise_class_probs = forbid_noise_class(char_noise_class_probs.squeeze(), []) # Forbids NOISE_SOS
        else:
            # in model
            char_noise_class_probs = torch.Tensor(TITLE_NOISE_PROBS).to(DEVICE)

        if i == title_format_length-1:
            if title_length == 0:
                char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_NONE,NOISE_REPLACE,NOISE_DELETE])
            elif title_length == 1:
                char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_DELETE])
            elif title_length == 2:
                char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_ADD])
            elif title_length == 3:
                char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_NONE, NOISE_REPLACE])
        if title_length == 4:
            char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_ADD, NOISE_DELETE])
        if title_length >= 5:
            char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_NONE,NOISE_ADD,NOISE_REPLACE])
        noise_class = NOISE[pyro.sample(f"{ADDRESS['title']}_{ADDRESS['noise']}_{i}", dist.Categorical(char_noise_class_probs))]
        noise_classes.append(noise_class)
        if noise_class == NOISE_NONE or noise_class == NOISE_REPLACE: title_length += 1
        if noise_class == NOISE_ADD: title_length += 2
    return title_length, noise_classes


def sample_name_noise(name_format_length: int, pyro_address: str, rnn: SequenceDenoisingModel = None, encoder_output: torch.Tensor = None) -> tuple:
    name_length, noise_classes = 0, []
    if rnn is not None:
        hidden_state = rnn.init_predictor_hidden()
        noise_type = NOISE_SOS
    
    for i in range(name_format_length):
        if rnn is not None:
            if i > 0: noise_type = noise_classes[-1]
            char_noise_class_probs, hidden_state = rnn.predict(noise_type, encoder_output[i], hidden_state)
            char_noise_class_probs = forbid_noise_class(char_noise_class_probs.squeeze(), []) # Forbids NOISE_SOS
        else:
            char_noise_class_probs = torch.Tensor(NAME_NOISE_PROBS).to(DEVICE)
        
        if i == name_format_length-1:
            if name_length == 0:
                char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_NONE,NOISE_REPLACE,NOISE_DELETE])
        if name_length == 9:
            char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_ADD])
        if name_length >= 10:
            char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_NONE,NOISE_ADD,NOISE_REPLACE])
        noise_class = NOISE[pyro.sample(f"{pyro_address}_{ADDRESS['noise']}_{i}", dist.Categorical(char_noise_class_probs))]
        noise_classes.append(noise_class)
        if noise_class == NOISE_NONE or noise_class == NOISE_REPLACE: name_length += 1
        if noise_class == NOISE_ADD: name_length += 2
    return name_length, noise_classes


def sample_suffix_noise(suffix_format_length, rnn: ClassificationDenoisingModel = None, encoder_output: torch.Tensor = None) -> tuple:
    suffix_length, noise_classes = 0, []    
    if rnn is not None:
        hidden_state = rnn.init_predictor_hidden()
        noise_type = NOISE_SOS
    
    for i in range(suffix_format_length):
        if rnn is not None:
            if i > 0: noise_type = noise_classes[-1]
            char_noise_class_probs, hidden_state = rnn.predict(noise_type, encoder_output[i], hidden_state)
            char_noise_class_probs = forbid_noise_class(char_noise_class_probs.squeeze(), []) # Forbids NOISE_SOS
        else:
            char_noise_class_probs = torch.Tensor(SUFFIX_NOISE_PROBS).to(DEVICE)
        
        if i == suffix_format_length-1:
            if suffix_length == 0:
                char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_NONE,NOISE_REPLACE,NOISE_DELETE])
        if suffix_length == 2:
            char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_ADD])
        if suffix_length >= 3:
            char_noise_class_probs = forbid_noise_class(char_noise_class_probs, [NOISE_NONE,NOISE_ADD,NOISE_REPLACE])
        noise_class = NOISE[pyro.sample(f"{ADDRESS['suffix']}_{ADDRESS['noise']}_{i}", dist.Categorical(char_noise_class_probs))]
        noise_classes.append(noise_class)
        if noise_class == NOISE_NONE or noise_class == NOISE_REPLACE: suffix_length += 1
        if noise_class == NOISE_ADD: suffix_length += 2
    return suffix_length, noise_classes
