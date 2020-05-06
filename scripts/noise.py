import pandas as pd
import random
import re
import string
import torch

from evaluate.eval_noiser import *

# Run from PDFAParser directory
# WARNING: If any parsed components have a separator character, the script will output UNKNOWN at name_noised field
WARNING_SYMBOL = 'UNKNOWN'
directory = 'data'
filenames = ['african', 'asian', 'british', 'hispanic']

# Apply 2 kinds of noise - component, separator
# Create one with 1 noise
# Have columns for 'noised components' to evaluate noisy parsing

def component_to_noise(series, components):
    noise_probs = torch.zeros(len(components))
    for i, name in enumerate(components):
        if components[i] != 'separator' and (series[name] != series[name] or len(series[name])<=1): continue
        noise_probs[i] = 1.
    component_index = torch.distributions.Categorical(noise_probs/noise_probs.sum()).sample().item()
    return component_index



for f in filenames:
    path = f"{directory}/{f}.csv"
    df = pd.read_csv(path)
    df['name_noised'] = float('nan')
    df['first_noised'] = float('nan')
    df['last_noised'] = float('nan')
    df['middle_noised'] = float('nan')
    df['title_noised'] = float('nan')
    df['suffix_noised'] = float('nan')

    # For each name
    #   Select an existing component to noise
    #   Populate noise columns
    #   Split the original full name string by space
    
    for i in range(len(df)):
        series = df.iloc[i]
        components = ['first', 'last', 'middle', 'title', 'suffix', 'separator']
        c_index = component_to_noise(series, components)

        error = False
        for j, c in enumerate(components):
            if c == 'separator': continue
            if j == c_index:
                series[f'{c}_noised'] = noise_name(series[c], [ch for ch in string.ascii_letters+string.digits+" ,"])
            else:
                series[f'{c}_noised'] = series[c]
        name_split = re.findall(r'([a-zA-Z\-\']*|,|\s|.)', series['name'])
        for j, c in enumerate(components):
            if c == 'separator': continue
            if series[c] != series[c]: continue
            if series[c] in name_split:
                name_split_index = name_split.index(series[c])
                name_split[name_split_index] = series[f'{c}_noised']
            else:
                error = True
        
        if error:
            series[f'name_noised'] = WARNING_SYMBOL
            df.iloc[i] = series
            continue
        
        if components[c_index] == 'separator':
            name_split = noise_separator(name_split)
        series[f'name_noised'] = ''.join(name_split)
        df.iloc[i] = series

    df.to_csv(f"{directory}/{f}.csv", index=False)
