import os
import pyro
import torch

from const import *
from pdfa.setup import generate_name_pdfa
from pdfa.symbol import *
from infcomp_helper import *
from neural_net.cc_model import CharacterClassificationModel
from neural_net.dae.denoiser import ClassificationDenoisingModel, SequenceDenoisingModel
from neural_net.pretrained.name_generator import NameGenerator
from utilities.config import *


class NameParser():
    def __init__(self, rnn_num_layers: int, format_rnn_hidden_size: int, dae_hidden_size: int, peak_prob: float = 1., noise_probs: list = None):
        super().__init__()

        self.peak_prob = peak_prob
        config = load_json('config/pretrained/first.json')
        self.output_chars = config['output']
        if noise_probs is None: noise_probs = [99/100, 1/300, 1/300, 1/300, 0.]
        self.noise_probs = noise_probs

        # Model neural nets instantiation
        self.model_fn = NameGenerator('config/pretrained/first.json', 'nn_model/pretrained/first.path.tar')
        self.model_ln = NameGenerator('config/pretrained/last.json', 'nn_model/pretrained/last.path.tar')

        # Guide neural nets instantiation
        self.guide_format = CharacterClassificationModel(PRINTABLE, format_rnn_hidden_size, SYMBOL, rnn_num_layers)
        self.guide_title = ClassificationDenoisingModel(PRINTABLE, dae_hidden_size, NOISE, TITLE_LIST, rnn_num_layers)
        self.guide_fn = SequenceDenoisingModel(PRINTABLE, dae_hidden_size, NOISE, self.output_chars,
                                               encoder_num_layers=rnn_num_layers)
        self.guide_ln = SequenceDenoisingModel(PRINTABLE, dae_hidden_size, NOISE, self.output_chars,
                                               encoder_num_layers=rnn_num_layers)
        self.guide_suffix = ClassificationDenoisingModel(PRINTABLE, dae_hidden_size, NOISE, SUFFIX_LIST,
                                                         encoder_num_layers=rnn_num_layers)

    def model(self, observations={"output": 0}):
        title, firstname, middlenames, lastname, suffix = '', '', '', '', ''
        title_obs_probs, firstname_obs_probs, middlenames_obs_probs, lastname_obs_probs, suffix_obs_probs = None, None, None, None, None

        with torch.no_grad():
            pdfa = generate_name_pdfa()
            name_format = sample_pdfa(pdfa)
            components = separate_components(name_format)

            if len(components[TITLE]) > 0:
                title, title_noise = sample_title_and_noise(components[TITLE], self.noise_probs)
                title_obs_probs = observation_probabilities(title, title_noise, self.peak_prob)
            if len(components[FIRST]) > 0:
                firstname, firstname_noise = sample_name_and_noise(components[FIRST], self.noise_probs, self.model_fn,
                                                                   ADDRESS['firstname'])
                firstname_obs_probs = observation_probabilities(firstname, firstname_noise, self.peak_prob)
            if len(components[MIDDLE]) > 0:
                middlenames, middlenames_noise = sample_multiple_name_and_noise(components[MIDDLE], self.noise_probs, self.model_fn,
                                                                                ADDRESS['middlename'])
                middlenames_obs_probs = []
                for middlename, middlename_noise in zip(middlenames, middlenames_noise):
                    middlename_obs_probs = observation_probabilities(middlename, middlename_noise, self.peak_prob)
                    middlenames_obs_probs.append(middlename_obs_probs)
            if len(components[LAST]) > 0:
                lastname, lastname_noise = sample_name_and_noise(components[LAST], self.noise_probs, self.model_ln, ADDRESS['lastname'])
                lastname_obs_probs = observation_probabilities(lastname, lastname_noise, self.peak_prob)
            if len(components[SUFFIX]) > 0:
                suffix, suffix_noise = sample_suffix_and_noise(components[SUFFIX], self.noise_probs)
                suffix_obs_probs = observation_probabilities(suffix, suffix_noise, self.peak_prob)

            fullname_obs_probs = combine_observation_probabilities(
                name_format=name_format,
                title=title_obs_probs,
                firstname=firstname_obs_probs,
                middlenames=middlenames_obs_probs,
                lastname=lastname_obs_probs,
                suffix=suffix_obs_probs,
                peak_prob=self.peak_prob
            )
            pyro.sample("output", pyro.distributions.Categorical(fullname_obs_probs), obs=observations["output"])

        return {
            'firstname': ''.join(firstname),
            'middlename': ' '.join([''.join(middlename) for middlename in middlenames]),
            'lastname': ''.join(lastname),
            'title': ''.join(title),
            'suffix': ''.join(suffix)
        }

    def guide(self, observations=None):
        observed = observations["output"]

        pyro.module("format", self.guide_format)
        pdfa = generate_name_pdfa()
        name_format = sample_conditional_pdfa(pdfa, self.guide_format, observed)
        name_parse = separate_name(name_format, observed)
        title, firstname, middlenames, lastname, suffix = '', '', '', '', ''

        if len(name_parse[TITLE]) > 0:
            pyro.module("title", self.guide_title)
            encoder_output, encoder_hidden = self.guide_title.encode(name_parse[TITLE][0])
            title, title_noise = sample_title_and_noise(name_parse[TITLE], self.noise_probs, self.guide_title, encoder_output,
                                                        encoder_hidden)
        if len(name_parse[FIRST]) > 0:
            pyro.module("first", self.guide_fn)
            encoder_output, encoder_hidden = self.guide_fn.encode(name_parse[FIRST][0])
            firstname, firstname_noise = sample_name_and_noise(name_parse[FIRST], self.noise_probs, self.guide_fn, ADDRESS['firstname'],
                                                               encoder_output, encoder_hidden)
        if len(name_parse[MIDDLE]) > 0:
            pyro.module("first", self.guide_fn)
            encoder_outputs = []
            encoder_hiddens = []
            for middle_parse in name_parse[MIDDLE]:
                encoder_output, encoder_hidden = self.guide_fn.encode(middle_parse)
                encoder_outputs.append(encoder_output)
                encoder_hiddens.append(encoder_hidden)
            middlenames, middlenames_noise = sample_multiple_name_and_noise(name_parse[MIDDLE], self.noise_probs, self.guide_fn,
                                                                            ADDRESS['middlename'], encoder_outputs,
                                                                            encoder_hiddens)
        if len(name_parse[LAST]) > 0:
            pyro.module("last", self.guide_ln)
            encoder_output, encoder_hidden = self.guide_ln.encode(name_parse[LAST][0])
            lastname, lastname_noise = sample_name_and_noise(name_parse[LAST], self.noise_probs, self.guide_ln, ADDRESS['lastname'],
                                                             encoder_output, encoder_hidden)
        if len(name_parse[SUFFIX]) > 0:
            pyro.module("suffix", self.guide_suffix)
            encoder_output, encoder_hidden = self.guide_suffix.encode(name_parse[SUFFIX][0])
            suffix, suffix_noise = sample_suffix_and_noise(name_parse[SUFFIX], self.noise_probs, self.guide_suffix, encoder_output,
                                                           encoder_hidden)
        result = {
            'firstname': ''.join(firstname),
            'middlename': ' '.join([''.join(middlename) for middlename in middlenames]),
            'lastname': ''.join(lastname),
            'title': ''.join(title),
            'suffix': ''.join(suffix)
        }
        return result

    def index_encode(self, name: str) -> torch.Tensor:
        if len(name) > MAX_STRING_LEN - 2: raise Exception(f"Name must be shorter than {MAX_STRING_LEN - 2}")
        result = torch.zeros(MAX_STRING_LEN).to(torch.long).to(DEVICE)
        name = [SOS] + list(name) + [EOS] + [PAD] * (MAX_STRING_LEN - len(name) - 2)
        for i in range(MAX_STRING_LEN):
            result[i] = PRINTABLE.index(name[i])
        return result
    
    def test_mode(self, noise_probs=None):
        # Call at inference time
        if noise_probs is None: noise_probs = [1-1e-4]+[1-(1-1e-4)/3]*3+[0.]
        self.noise_probs = noise_probs
        self.guide_format.test_mode()
        self.guide_title.test_mode()
        self.guide_fn.test_mode()
        self.guide_ln.test_mode()
        self.guide_suffix.test_mode()
    
    def load_checkpoint(self, folder="nn_model", filename="checkpoint"):
        aux_fp = os.path.join(folder, f"{filename}_aux.pth.tar")
        name_fp = os.path.join(folder, f"{filename}_name.pth.tar")
        if not os.path.exists(aux_fp):
            raise Exception(f"Weights {aux_fp} does not exist.")
        if not os.path.exists(name_fp):
            raise Exception(f"Weights {name_fp} does not exist.")
        aux_content = torch.load(aux_fp, map_location=DEVICE)
        name_content = torch.load(name_fp, map_location=DEVICE)
        # name content
        self.guide_fn.load_state_dict(name_content['guide_fn'])
        self.guide_ln.load_state_dict(name_content['guide_ln'])
        # title and suffix
        self.guide_title.load_state_dict(aux_content['guide_title'])
        self.guide_suffix.load_state_dict(aux_content['guide_suffix'])
        # format content
        self.guide_format.load_state_dict(aux_content['guide_format'])

    def save_checkpoint(self, folder="nn_model", filename="checkpoint"):
        if not os.path.exists(folder): os.mkdir(folder)
        aux_fp = os.path.join(folder, f"{filename}_aux.pth.tar")
        name_fp = os.path.join(folder, f"{filename}_name.pth.tar")
        aux_content = {
            'guide_format': self.guide_format.state_dict(),
            'guide_title': self.guide_title.state_dict(),
            'guide_suffix': self.guide_suffix.state_dict(),
        }
        name_content = {
            'guide_fn': self.guide_fn.state_dict(),
            'guide_ln': self.guide_ln.state_dict()
        }
        torch.save(aux_content, aux_fp)
        torch.save(name_content, name_fp)
