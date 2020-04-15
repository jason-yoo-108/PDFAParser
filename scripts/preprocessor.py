import ast
import pandas as pd

df = pd.read_csv('full_name_source.csv')[:5000]

names = []
for i, movie in df.iterrows():
    if i%100 == 0: print(f"Iteration {i}")
    casts = ast.literal_eval(movie['cast'])
    for cast in casts:
        names.append(cast['name'])
    #crews = ast.literal_eval(movie['crew'])
    #for crew in crews:
    #    names.append(crew['name'])
import re
def clean_regex_match(x):
    regex = "^ {0,4}((Mr|Ms|Mrs|Sir|Madam|Dr).? {0,4})?((\w|-|'){1,10}.? {0,4}((\w|-|'){1,10}.? {0,4}){0,2}(\w|-|'){1,10}.?|(\w|-|'){1,10}.?, {0,4}(\w|-|'){1,10}.?( {0,4}(\w|-|'){1,10}.?){0,2})( {0,4}(Jr|Jnr|Sr|Snr|PhD|MD|I|II|III|IV).?)? {0,4}$"
    return re.search(regex, x)

def filter_function(x):
    isascii = lambda s: len(s) == len(s.encode())
    if len(x) != len(x.encode()): return False
    if len(x) > 48: return False
    split = x.split(' ')
    for n in split:
        if len(n) > 10: return False
    if len(split) == 1: return False
    if x.count("'") > 1: return False
    if ',' in x or '"' in x: return False
    return True

clean_names = list(set(filter(lambda name: filter_function(name) and clean_regex_match(name), names)))

clean_df = pd.DataFrame({'name': clean_names})[:1000]
clean_df.to_csv('fullnames.csv', index=False)
