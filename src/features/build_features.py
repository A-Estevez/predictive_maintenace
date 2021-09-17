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
predictive_cols = cols[0:7]