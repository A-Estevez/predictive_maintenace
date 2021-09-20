#%%
import pandas as pd

#%%
df = pd.read_csv('data/raw/ai4i2020.csv')
print(df.head(5))
#%%
cols = df.columns

# %%
target = df[cols[8]]

# %%
predictive_cols = cols[1:7]
#%%
temp = df[predictive_cols[2]]
past_air_t = []
for idx,val in enumerate(temp):
    too_hot = sum(idx > i and temp[idx-i] > 299 for i in range(1, 6))
    past_air_t.append(too_hot)

#%%
df['past_air_t'] = past_air_t
#%%

col = df[predictive_cols[3]]
past_positives = []
for idx,val in enumerate(col):
    positives = sum(idx > i and col[idx-i] > 309 for i in range(1, 6))
    past_positives.append(positives)

#%%
df['past_proces_t'] = past_positives
#%%

col = df[predictive_cols[4]]
past_positives = []
for idx,val in enumerate(col):
    positives = sum(idx > i and col[idx-i] > 2000 for i in range(1, 6))
    past_positives.append(positives)

#%%
df['past_rotational'] = past_positives
#%%

col = df[predictive_cols[5]]
past_positives = []
for idx,val in enumerate(col):
    positives = sum(idx > i and col[idx-i] > 50 for i in range(1, 6))
    past_positives.append(positives)

#%%
df['past_torque'] = past_positives
#%%
col_n_list = range(2,6)
new_col_name_list = ['past_air_t', 'past_proces_t', 'past_rotational', 'past_torque']
umbral_value_list = [299, 309, 2000, 50]

def past_value_checker(col_n, umbral_value, new_col_name):
    col = df[predictive_cols[col_n]]
    past_positives = []
    for idx,val in enumerate(col):
        positives = sum(idx > i and col[idx-i] > umbral_value for i in range(1, 6))
        past_positives.append(positives)
    df[new_col_name] = past_positives
#%%
for col_n, umbral_value, new_col_name in col_n_list, umbral_value_list, new_col_name_list:
    past_value_checker(col_n, umbral_value, new_col_name)
#%%
print(df.columns)
#%%
generated_vars = df.columns[14:]