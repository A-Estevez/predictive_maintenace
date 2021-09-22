import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%%
df = pd.read_csv('data/interim/features.csv')

#%%
column_names = df.columns
print(column_names)

#%%
target = 'Machine failure'

#%%


#correlation heatmap (Pearson)
#replace "df" with your df
columns_drop = ['UDI']#columns we don't want to use
plt.figure(figsize=(12, 6))
df_to_plot = df.copy()
df_to_plot.drop(
        columns_drop, axis=1, inplace=True
        )
corr = df_to_plot.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(
    corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.2,
    cmap='YlGnBu'
    )

#%%
from scipy import stats
#define resume tables
def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=10),2) 

    return summary

#%%
resume = resumetable(df)