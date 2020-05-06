"""
Cite: @inproceedings{ambekar2009name, title={Name-ethnicity classification from open sources}, author={Ambekar, Anurag and Ward, Charles and Mohammed, Jahangir and Male, Swapna and Skiena, Steven}, booktitle={Proceedings of the 15th ACM SIGKDD international conference on Knowledge Discovery and Data Mining}, pages={49--58}, year={2009}, organization={ACM} }
Paper: https://dl.acm.org/doi/10.1145/1557019.1557032

Take 250 Greater African Names
Take 250 Greater Asian Names
Take 250 Greater British Names
Take 250 Hispanic Names

Take 100 Names with Middle Names

Take 10 Title
Take 10 Suffix
"""

import pandas as pd

SEED = 1
NUM_NAMES_WITH_MIDDLE = 110
NUM_NAMES_WITHOUT_MIDDLE = 160
NUM_NAMES_WITH_SUFFIX = 20
MAX_COMPONENT_LENGTH = 10


# Load and Shuffle
df = pd.read_csv('wiki_name_race.csv')
df = df.sample(frac=1,random_state=SEED).reset_index(drop=True)

# Remove names that are longer than length 10
# Remove all characters other than a-zA-Z'-
# Capitalize all names
def filter_on_middlename(s):
    if s != s: return True
    split = s.split(' ')
    for c in split:
        if len(c)>MAX_COMPONENT_LENGTH: return False
    return True
length_mask = (df['name_last'].str.len() <= MAX_COMPONENT_LENGTH) & (df['name_first'].str.len() <= MAX_COMPONENT_LENGTH) & ((df['name_middle'].isnull()) | (df['name_middle'].str.len() <= MAX_COMPONENT_LENGTH*2))
df = df.loc[length_mask]
df = df[df['name_middle'].map(filter_on_middlename)]
df['name_first'] = df['name_first'].apply(lambda x: float('nan') if isinstance(x,float) or len(x.encode()) != len(x) else x)
df['name_middle'] = df['name_middle'].apply(lambda x: float('nan') if isinstance(x,float) or len(x.encode()) != len(x) else x)
df['name_last'] = df['name_last'].apply(lambda x: float('nan') if isinstance(x,float) or len(x.encode()) != len(x) else x)
df.dropna(subset=['name_first','name_last'], inplace=True)


df['name_first'] = df['name_first'].apply(lambda x: ''.join(["" if (ord(i) < 65 or ord(i) > 122) and ord(i) not in [32,39,45] else i for i in x]))
df['name_middle'] = df['name_middle'].apply(lambda x: ''.join(["" if (ord(i) < 65 or ord(i) > 122) and ord(i) not in [32,39,45] else i for i in x]) if isinstance(x, str) else x)
df['name_last'] = df['name_last'].apply(lambda x: ''.join(["" if (ord(i) < 65 or ord(i) > 122) and ord(i) not in [32,39,45] else i for i in x]))
df['name_first'] = df['name_first'].apply(lambda x: x.title())
df['name_middle'] = df['name_middle'].apply(lambda x: x.title() if not isinstance(x,float) else x)
df['name_last'] = df['name_last'].apply(lambda x: x.title())
def suffix_cap(x):
    if isinstance(x,float): return x
    if 'i' in x: return x.upper()
    return x.capitalize()
df['name_suffix'] = df['name_suffix'].apply(suffix_cap)



df_african = df[df['race'].str.match('GreaterAfrican,African')]
df_asian = df[df['race'].str.match('Asian*')]
df_british = df[df['race'].str.match('GreaterEuropean,British')]
df_hispanic = df[df['race'].str.match('GreaterEuropean,WestEuropean,Hispanic')]
race_dfs = [df_african, df_asian, df_british, df_hispanic]

race_dfs_new = []
for j, rd in enumerate(race_dfs):
    print(f"Processing DataFrame {j}...")
    rd['name_full'] = float('nan')
    rd['name_title'] = float('nan')
    race_df = rd[rd['name_suffix'].isnull()]

    w_middle = race_df.loc[~race_df['name_middle'].isnull()]
    for i, r in enumerate(w_middle.iterrows()):
        row = r[1]
        if i%2 == 0:
            w_middle.loc[:,'name_full'][r[0]] = f"{row['name_first']} {row['name_middle']} {row['name_last']}"
        else:
            w_middle.loc[:,'name_full'][r[0]] = f"{row['name_last']}, {row['name_first']} {row['name_middle']}"

    wo_middle = race_df.loc[race_df['name_middle'].isnull()]
    for i, r in enumerate(wo_middle.iterrows()):
        row = r[1]
        if i%2 == 0:
            wo_middle.loc[:,'name_full'][r[0]] = f"{row['name_first']} {row['name_last']}"
        else:
            wo_middle.loc[:,'name_full'][r[0]] = f"{row['name_last']}, {row['name_first']}"

    w_suffix = rd.loc[~rd['name_suffix'].isnull()]
    for i, r in enumerate(w_suffix.iterrows()):
        row = r[1]
        if row['name_middle'] == row['name_middle']:
            tmp = f"{row['name_first']} {row['name_middle']} {row['name_last']}"
        else:
            tmp = f"{row['name_first']} {row['name_last']}"
        w_suffix.loc[:,'name_full'][r[0]] = f"{tmp} {row['name_suffix'].replace('.','')}"

    joined_df = pd.concat([w_middle[:NUM_NAMES_WITH_MIDDLE], wo_middle[:NUM_NAMES_WITHOUT_MIDDLE], w_suffix[:NUM_NAMES_WITH_SUFFIX]])
    joined_df.rename(columns={'name_full': 'name', 'name_first': 'first', 'name_middle': 'middle', 'name_last': 'last', 'name_title': 'title', 'name_suffix': 'suffix'}, inplace=True)
    race_dfs_new.append(joined_df[['name','first','last','middle','title','suffix']])



race_dfs_new[0].to_csv('african.csv', index=False)
race_dfs_new[1].to_csv('asian.csv', index=False)
race_dfs_new[2].to_csv('british.csv', index=False)
race_dfs_new[3].to_csv('hispanic.csv', index=False)
