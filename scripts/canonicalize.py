import pandas as pd

PATH = 'result/entries/4.2african_noised_full'

correct_df = pd.read_csv(f"{PATH}_correct.csv")
incorrect_df = pd.read_csv(f"{PATH}_incorrect.csv")

def get_canonicalized_names(df):
    canonicalized = []
    for i, row in df.iterrows():
        title, first, middle, last, suffix = row['predicted title'], row['predicted first'], row['predicted middle'], row['predicted last'], row['predicted suffix']
        result = ''
        if title == title: result += title+' '
        if first == first: result += first+' '
        if middle == middle: result += middle+' '
        if last == last: result += last+' '
        if suffix == suffix: result += suffix
        if i%5 == 0: print(f"{row['original']} & {row['noised']} & {result}\\\\")
        canonicalized.append(result)

    return canonicalized

print("CORRECT")
correct_df['canonicalized'] = get_canonicalized_names(correct_df)
print("INCORRECT")
incorrect_df['canonicalized'] = get_canonicalized_names(incorrect_df)

#correct_df.to_csv(f'{PATH}_correct.csv')
#incorrect_df.to_csv(f'{PATH}_incorrect.csv')
