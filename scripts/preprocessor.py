import ast
import pandas as pd

df = pd.read_csv('fullnames.csv')[:5000]

names = df['name'].tolist()

titles, firsts, middles, lasts, suffixs = [], [], [], [], []
for name in names:
    title, first, middle, last, suffix = '', '', '', '', ''
    split = name.split(' ')
    if len(split) == 2:
        first = split[0]
        last = split[1]
    if len(split) == 3:
        first = split[0]
        middle = split[1] if '.' not in split[1] else split[1][:-1]
        last = split[2]
    if len(split) == 4:
        first = split[0]
        middle = split[1]+' '+split[2]
        last = split[3]
    titles.append(title)
    firsts.append(first)
    middles.append(middle)
    lasts.append(last)
    suffixs.append(suffix)

parse_df = pd.DataFrame({'name': names, 'first': firsts, 'last': lasts, 'middle': middles, 'title': titles, 'suffix': suffixs})
parse_df.to_csv('test.csv', index=False)
