#%%
import pandas as pd
import numpy as np

#%%
df = pd.read_csv('data/raw/ai4i2020.csv')
print(df.head(5))
#%%
cols = df.columns

# %%
target = df[cols[8]]

#%%
error_tipes_to_remove = [9, 10, 11, 12, 13]
#%%
df.drop(df.columns[error_tipes_to_remove], axis = 1, inplace = True)

# %%
predictive_cols = cols[2:7]
#%%
# df.iloc[[0, 2], [1, 3]]
df_error_type = df.iloc[:,9:]
df = df.iloc[1:,:9]


#%%
col_n_list = range(1,5)
new_col_name_list = ['past_air_t', 'past_proces_t', 'past_rotational', 'past_torque']
umbral_value_list = [299, 309, 2000, 50]

#%%

def past_value_checker(col_n, umbral_value, new_col_name):
    col = df[predictive_cols[col_n]]
    past_positives = []
    for idx,val in enumerate(col):
        positives = sum(idx > i and col[idx-i] > umbral_value for i in range(1, 6))
        past_positives.append(positives)
    df[new_col_name] = past_positives
#%%
for col_n, umbral_value, new_col_name in zip(col_n_list, umbral_value_list, new_col_name_list):
    past_value_checker(col_n, umbral_value, new_col_name)
#%%
generated_vars = df.columns[14:]

#%%
to_cat_num = df[predictive_cols[0]]
from statsmodels.tools import categorical
a = np.array(to_cat_num.values.tolist())
b = categorical(a, drop=True)
type_replace = np.argmax(b, axis = 1)
#%%
df['type_num'] = type_replace

#%%
df.drop('Type', axis = 1, inplace = True)
df.drop('Product ID', axis = 1, inplace = True)

#%%
df.to_csv(r'data/interim/features.csv', index = False)
df_error_type.to_csv(r'data/interim/error_type.csv', index = False)