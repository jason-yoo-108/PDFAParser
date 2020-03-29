import string
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS = 'SOS'  # '<SOS>'
EOS = 'EOS'  # '<EOS>'
PAD = 'PAD'  # '<PAD>'

# Constants just for FORMAT_CLASS
SPACE = ' '
DOT = '.'
COMMA = ','

# for infcomp.py
MAX_OUTPUT_LEN = 10
MAX_STRING_LEN = 35
PRINTABLE = [char for char in string.printable] + [SOS, PAD, EOS]
NUM_PRINTABLE = len(PRINTABLE)
