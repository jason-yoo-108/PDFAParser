# Symbols: All symbols that can be emitted by a PDFA state
# Name Components
TITLE = 'T'  # 'TITLE'
FIRST = 'F'  # 'FIRST'
MIDDLE = 'M'  # 'MIDDLE'
LAST = 'L'  # 'LAST'
SUFFIX = 'S'  # 'SUFFIX'
# Separators
SPACE = ' '  # 'SPACE'
COMMA = ','  # 'COMMA'
PERIOD = '.'  # 'PERIOD'
# Misc
EOS_FORMAT = '1'  # 'EOS'
PAD_FORMAT = '2'  # 'PAD'
SOS_FORMAT = '3'  # 'SOS
SYMBOL = [
    TITLE, FIRST, MIDDLE, LAST, SUFFIX, SPACE,
    COMMA, PERIOD, EOS_FORMAT, PAD_FORMAT, SOS_FORMAT
]
