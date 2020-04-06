import torch

from const import PRINTABLE, NUM_PRINTABLE

CHARACTER_REPLACEMENT = dict()
CHARACTER_REPLACEMENT['A'] = 'QSZWXa'
CHARACTER_REPLACEMENT['B'] = 'NHGVb '
CHARACTER_REPLACEMENT['C'] = 'VFDXc '
CHARACTER_REPLACEMENT['D'] = 'FRESXCd'
CHARACTER_REPLACEMENT['E'] = 'SDFR$#WSe'
CHARACTER_REPLACEMENT['F'] = 'GTRDCVf'
CHARACTER_REPLACEMENT['G'] = 'HYTFVBg'
CHARACTER_REPLACEMENT['H'] = 'JUYTGBNh'
CHARACTER_REPLACEMENT['I'] = 'UJKLO(*i'
CHARACTER_REPLACEMENT['J'] = 'MKIUYHNj'
CHARACTER_REPLACEMENT['K'] = 'JM<LOIk'
CHARACTER_REPLACEMENT['L'] = 'K<>:POl'
CHARACTER_REPLACEMENT['M'] = 'NJK< m'
CHARACTER_REPLACEMENT['N'] = 'BHJM n'
CHARACTER_REPLACEMENT['O'] = 'PLKI()Po'
CHARACTER_REPLACEMENT['P'] = 'OL:{_)O"p'
CHARACTER_REPLACEMENT['Q'] = 'ASW@!q'
CHARACTER_REPLACEMENT['R'] = 'TFDE$r%'
CHARACTER_REPLACEMENT['S'] = 'DXZAWEs'
CHARACTER_REPLACEMENT['T'] = 'YGFR%^t'
CHARACTER_REPLACEMENT['U'] = 'IJHY&*u'
CHARACTER_REPLACEMENT['V'] = 'CFGB v'
CHARACTER_REPLACEMENT['W'] = 'SAQ@#Ew'
CHARACTER_REPLACEMENT['X'] = 'ZASDCx'
CHARACTER_REPLACEMENT['Y'] = 'UGHT^&y'
CHARACTER_REPLACEMENT['Z'] = 'XSAz'
CHARACTER_REPLACEMENT['a'] = 'qwszA'
CHARACTER_REPLACEMENT['b'] = 'nhgv B'
CHARACTER_REPLACEMENT['c'] = 'vfdx C'
CHARACTER_REPLACEMENT['d'] = 'fresxcD'
CHARACTER_REPLACEMENT['e'] = 'sdfr43wsE'
CHARACTER_REPLACEMENT['f'] = 'gtrdcvF'
CHARACTER_REPLACEMENT['g'] = 'hytfvbG'
CHARACTER_REPLACEMENT['h'] = 'juytgbnH'
CHARACTER_REPLACEMENT['i'] = 'ujklo98I'
CHARACTER_REPLACEMENT['j'] = 'mkiuyhnJ'
CHARACTER_REPLACEMENT['k'] = 'jm,loijK'
CHARACTER_REPLACEMENT['l'] = 'k,.;pokL'
CHARACTER_REPLACEMENT['m'] = 'njk, M'
CHARACTER_REPLACEMENT['n'] = 'bhjm N'
CHARACTER_REPLACEMENT['o'] = 'plki90pO'
CHARACTER_REPLACEMENT['p'] = 'ol;[-0oP'
CHARACTER_REPLACEMENT['q'] = 'asw21Q'
CHARACTER_REPLACEMENT['r'] = 'tfde45R'
CHARACTER_REPLACEMENT['s'] = 'dxzaweS'
CHARACTER_REPLACEMENT['t'] = 'ygfr56T'
CHARACTER_REPLACEMENT['u'] = 'ijhy78U'
CHARACTER_REPLACEMENT['v'] = 'cfgb V'
CHARACTER_REPLACEMENT['w'] = 'saq23eW'
CHARACTER_REPLACEMENT['x'] = 'zsdcX'
CHARACTER_REPLACEMENT['y'] = 'uhgt67Y'
CHARACTER_REPLACEMENT['z'] = 'xsaZ'
CHARACTER_REPLACEMENT['1'] = '2q'
CHARACTER_REPLACEMENT['2'] = '3wq1'
CHARACTER_REPLACEMENT['3'] = '4ew2'
CHARACTER_REPLACEMENT['4'] = '5re3'
CHARACTER_REPLACEMENT['5'] = '6tr4'
CHARACTER_REPLACEMENT['6'] = '7yt5'
CHARACTER_REPLACEMENT['7'] = '8uy6'
CHARACTER_REPLACEMENT['8'] = '9iu7'
CHARACTER_REPLACEMENT['9'] = '0oi8'
CHARACTER_REPLACEMENT['0'] = '-po9'


def insert_peaked_probs(char: str, peak_prob: float) -> list:
    char_prob = [(1 - peak_prob) / (NUM_PRINTABLE - 1)] * NUM_PRINTABLE
    char_prob[PRINTABLE.index(char)] = peak_prob
    return char_prob


def insert_noise_probs(character: str, nearby_char_total_prob: float = .8) -> list:
    char_prob = []

    if character in CHARACTER_REPLACEMENT:
        nearby_chars = CHARACTER_REPLACEMENT[character]
        nearby_chars_len = len(nearby_chars)

        nearby_char_prob = nearby_char_total_prob / nearby_chars_len
        non_near_chars_total_prob = 1 - nearby_char_total_prob
        non_near_char_prob = non_near_chars_total_prob / (NUM_PRINTABLE - nearby_chars_len)

        char_prob = [0] * NUM_PRINTABLE
        for i in range(NUM_PRINTABLE):
            if PRINTABLE[i] in nearby_chars:
                char_prob[i] = nearby_char_prob
            else:
                char_prob[i] = non_near_char_prob
    else:
        char_prob = [1 / NUM_PRINTABLE] * NUM_PRINTABLE

    return char_prob