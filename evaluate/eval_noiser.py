from random import randint

import torch
import torch.distributions as distributions

CHARACTER_REPLACEMENT = dict()
CHARACTER_REPLACEMENT['A'] = 'QSZWXa'
CHARACTER_REPLACEMENT['B'] = 'NHGVb'
CHARACTER_REPLACEMENT['C'] = 'VFDXc'
CHARACTER_REPLACEMENT['D'] = 'FRESXCd'
CHARACTER_REPLACEMENT['E'] = 'SDFR$#WSe'
CHARACTER_REPLACEMENT['F'] = 'GTRDCVf'
CHARACTER_REPLACEMENT['G'] = 'HYTFVBg'
CHARACTER_REPLACEMENT['H'] = 'JUYTGBNh'
CHARACTER_REPLACEMENT['I'] = 'UJKLO(*i'
CHARACTER_REPLACEMENT['J'] = 'MKIUYHNj'
CHARACTER_REPLACEMENT['K'] = 'JM<LOIk'
CHARACTER_REPLACEMENT['L'] = 'K<>:POl'
CHARACTER_REPLACEMENT['M'] = 'NJK<m'
CHARACTER_REPLACEMENT['N'] = 'BHJMn'
CHARACTER_REPLACEMENT['O'] = 'PLKI()Po'
CHARACTER_REPLACEMENT['P'] = 'OL:{_)O"p'
CHARACTER_REPLACEMENT['Q'] = 'ASW@!q'
CHARACTER_REPLACEMENT['R'] = 'TFDE$r%'
CHARACTER_REPLACEMENT['S'] = 'DXZAWEs'
CHARACTER_REPLACEMENT['T'] = 'YGFR%^t'
CHARACTER_REPLACEMENT['U'] = 'IJHY&*u'
CHARACTER_REPLACEMENT['V'] = 'CFGBv'
CHARACTER_REPLACEMENT['W'] = 'SAQ@#Ew'
CHARACTER_REPLACEMENT['X'] = 'ZASDCx'
CHARACTER_REPLACEMENT['Y'] = 'UGHT^&y'
CHARACTER_REPLACEMENT['Z'] = 'XSAz'
CHARACTER_REPLACEMENT['a'] = 'qwszA'
CHARACTER_REPLACEMENT['b'] = 'nhgvB'
CHARACTER_REPLACEMENT['c'] = 'vfdxC'
CHARACTER_REPLACEMENT['d'] = 'fresxcD'
CHARACTER_REPLACEMENT['e'] = 'sdfr43wsE'
CHARACTER_REPLACEMENT['f'] = 'gtrdcvF'
CHARACTER_REPLACEMENT['g'] = 'hytfvbG'
CHARACTER_REPLACEMENT['h'] = 'juytgbnH'
CHARACTER_REPLACEMENT['i'] = 'ujklo98I'
CHARACTER_REPLACEMENT['j'] = 'mkiuyhnJ'
CHARACTER_REPLACEMENT['k'] = 'jm,loijK'
CHARACTER_REPLACEMENT['l'] = 'k,.;pokL'
CHARACTER_REPLACEMENT['m'] = 'njk,M'
CHARACTER_REPLACEMENT['n'] = 'bhjmN'
CHARACTER_REPLACEMENT['o'] = 'plki90pO'
CHARACTER_REPLACEMENT['p'] = 'ol;[-0oP'
CHARACTER_REPLACEMENT['q'] = 'asw21Q'
CHARACTER_REPLACEMENT['r'] = 'tfde45R'
CHARACTER_REPLACEMENT['s'] = 'dxzaweS'
CHARACTER_REPLACEMENT['t'] = 'ygfr56T'
CHARACTER_REPLACEMENT['u'] = 'ijhy78U'
CHARACTER_REPLACEMENT['v'] = 'cfgbV'
CHARACTER_REPLACEMENT['w'] = 'saq23eW'
CHARACTER_REPLACEMENT['x'] = 'zsdcX'
CHARACTER_REPLACEMENT['y'] = 'uhgt67Y'
CHARACTER_REPLACEMENT['z'] = 'xsaZ'
CHARACTER_REPLACEMENT['1'] = '2q~`'
CHARACTER_REPLACEMENT['2'] = '3wq1'
CHARACTER_REPLACEMENT['3'] = '4ew2'
CHARACTER_REPLACEMENT['4'] = '5re3'
CHARACTER_REPLACEMENT['5'] = '6tr4'
CHARACTER_REPLACEMENT['6'] = '7yt5'
CHARACTER_REPLACEMENT['7'] = '8uy6'
CHARACTER_REPLACEMENT['8'] = '9iu7'
CHARACTER_REPLACEMENT['9'] = '0oi8'
CHARACTER_REPLACEMENT['0'] = '-po9'
CHARACTER_REPLACEMENT['-'] = '_=+~'
CHARACTER_REPLACEMENT['.'] = ',\';`'
CHARACTER_REPLACEMENT['\''] = '"`'


def noise_name(x: str, allowed_chars: str, max_noise: int = 1):
    noise_type = distributions.Categorical(torch.tensor([1/3, 1/6, 1/3, 1/6])).sample().item()
    x_length = len(x)
    ret = x
    if noise_type == 0:
        ret = add_chars(x, allowed_chars, max_add=max_noise)
    elif noise_type == 1:
        ret = switch_chars(x, allowed_chars, max_switch=max_noise)
    elif noise_type == 2 and x_length != 1:
        ret = remove_chars(x, max_remove=max_noise)
    elif noise_type == 3:
        ret = switch_to_similar(x, allowed_chars, max_switch=max_noise)
    return ret


def add_chars(x: str, allowed_chars: str, max_add: int):
    ret = x
    for i in range(max_add):
        random_char = allowed_chars[randint(0, len(allowed_chars) - 1)]
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], random_char, ret[pos:]))
    return ret


def switch_chars(x: str, allowed_chars: str, max_switch: int):
    if len(x) < max_switch:
        max_switch = len(x)

    ret = x
    for i in range(max_switch):
        pos = distributions.Categorical(torch.tensor([1 / len(x)] * len(x))).sample().item()
        random_char = allowed_chars[
            distributions.Categorical(torch.tensor([1 / len(allowed_chars)] * len(allowed_chars))).sample().item()]
        ret = "".join((ret[:pos], random_char, ret[pos + 1:]))
    return ret


def switch_to_similar(x: str, allowed_chars: str, max_switch: int):
    if len(x) < max_switch:
        max_switch = len(x)

    ret = x
    for i in range(max_switch):
        pos = distributions.Categorical(torch.tensor([1 / len(x)] * len(x))).sample().item()
        current_char = x[pos]
        replacements = CHARACTER_REPLACEMENT[x[pos]] if x[pos] in CHARACTER_REPLACEMENT else allowed_chars
        random_char = replacements[
            distributions.Categorical(torch.tensor([1 / len(replacements)] * len(replacements))).sample().item()]
        ret = "".join((ret[:pos], random_char, ret[pos + 1:]))
    return ret


def remove_chars(x: str, max_remove: int):
    if len(x) <= max_remove:
        max_remove = len(x) - 1

    ret = x
    for i in range(max_remove):
        x_length = len(ret)
        pos = distributions.Categorical(torch.tensor([1 / x_length] * x_length)).sample().item()
        ret = "".join((ret[:pos], ret[pos + 1:]))
    return ret


def noise_separator(name_split: list):
    recognized_separators = [',',' ','.']
    noise_probs = torch.zeros(len(name_split))
    for i, name in enumerate(name_split):
        if name_split[i] in recognized_separators:
            noise_probs[i] = 1.
    noise_index = torch.distributions.Categorical(noise_probs/noise_probs.sum()).sample().item()
    separator = name_split[noise_index]

    # Type of noise to apply
    noise_type = torch.distributions.Categorical(torch.Tensor([1/3]*3)).sample().item()
    if noise_type == 0: # Add a character after
        random_addition = separator
        name_split[noise_index] = separator + random_addition
    elif noise_type == 1: # Add a random character
        replace_probs = torch.zeros(len(recognized_separators))
        for i, char in enumerate(recognized_separators):
            if separator == char: continue
            replace_probs[i] = 1.
        random_replacement = recognized_separators[torch.distributions.Categorical(replace_probs/replace_probs.sum()).sample().item()]
        name_split[noise_index] = random_replacement
    else:
        name_split[noise_index] = ''
    return name_split
