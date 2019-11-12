#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:43:12 2019

@author: Heitor
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set_style("white")
sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 2.5})

# Read in csv file
df = pd.read_csv('Spec41.csv')



names=[]
n=0
for i in df['Sp']:
    names.append(i+' '+str(df['line'][n]))
    n=n+1

df1 = pd.DataFrame()
df1['line'] = names

df = df.assign(line=df1)

#df = df.drop(["Sp"], axis = 1)
df = df.drop(["Sp", "line"], axis = 1)



# cmap is now a list of colors
cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=2, reverse=True)


# Create two appropriately sized subplots
grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}
fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws)

ax = sns.heatmap(df, ax=ax, cbar_ax=cbar_ax, cmap=ListedColormap(cmap),
                 linewidths=.5, linecolor='lightgray',
                 cbar_kws={'orientation': 'vertical'} , yticklabels=True)

# Customize tick marks and positions
cbar_ax.set_yticklabels(['B', 'G'])
cbar_ax.yaxis.set_ticks([ 0.16666667, 0.83333333])


# X - Y axis labels
ax.set_ylabel('Lines')
ax.set_xlabel('[Fe/H]')
ax.set(yticks=range(0, 74), yticklabels=names, xticklabels=['-3.0 20k','-3.0 40k','-1.0 20k','-1.0 40k'])
ax.set_title('Giant')

# Rotate tick labels
locs, labels = plt.xticks()
plt.setp(labels, rotation=0)
locs, labels = plt.yticks()
plt.setp(labels, rotation=0)





















#