import string

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS = 'SOS'  # '<SOS>'
EOS = 'EOS'  # '<EOS>'
PAD = 'PAD'  # '<PAD>'

# for infcomp.py
MAX_OUTPUT_LEN = 10
MAX_STRING_LEN = 50
PRINTABLE = [char for char in string.printable] + [SOS, PAD, EOS]
NUM_PRINTABLE = len(PRINTABLE)

ADDRESS = {
    'format': 'format',
    'noise': 'noise',
    'title': 'title',
    'firstname': 'firstname',
    'middlename': 'middlename',
    'lastname': 'lastname',
    'suffix': 'suffix'
}

# Noising Probabilities: [<none>, <add a character>, <replace a character>, <remove a character>]
NOISE_NONE = 'X'
NOISE_ADD = 'A'
NOISE_REPLACE = 'R'
NOISE_DELETE = 'D'
NOISE_SOS = '1'
NOISE = [NOISE_NONE, NOISE_ADD, NOISE_REPLACE, NOISE_DELETE, NOISE_SOS]

TITLE_LIST = ['Mr', 'Ms', 'Dr', 'Mrs', 'Sir', "Ma'am", 'Madam']
# Title Probabilities Per Length
TITLE_2 = {'Mr': (0, 0.45), 'Ms': (1, 0.45), 'Dr': (2, 0.1)}
TITLE_3 = {'Mrs': (3, 0.9), 'Sir': (4, 0.1)}
TITLE_5 = {"Ma'am": (5, 0.5), 'Madam': (6, 0.5)}

SUFFIX_LIST = ['Sr', 'Snr', 'Jr', 'Jnr', 'PhD', 'MD', 'I', 'II', 'III', 'IV']
# Suffix Probabilities Per Length
# Length 1: [I]
SUFFIX_1 = {'I': (6, 1.)}
# Length 2: [Sr, Jr, MD, II, IV]
SUFFIX_2 = {'Sr': (0, 0.33), 'Jr': (2, 0.33), 'MD': (5, 0.33), 'II': (7, 0.005), 'IV': (9, 0.005)}
# Length 3: [Snr, Jnr, Phd, III]
SUFFIX_3 = {'Snr': (1, 0.33), 'Jnr': (3, 0.33), 'PhD': (4, 0.33), 'III': (8, 0.01)}
